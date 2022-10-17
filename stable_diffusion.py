from typing import List, Union
from enum import Enum, auto

import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
)
from diffusers.models.attention import BasicTransformerBlock
from tqdm.auto import tqdm
from time import time


class Scheduler(Enum):
    DDIM = auto()
    PNDM = auto()
    LMSD = auto()


class StableDiffusion:
    def __init__(
        self,
        weights: str = "CompVis/stable-diffusion-v1-4", 
        scheduler: Scheduler = Scheduler.PNDM,
        torch_device: Union[None, str] = None,
    ) -> None:
        # remove warning when loading models
        logging.set_verbosity_error()

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(weights, subfolder="vae", use_auth_token=True)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        clip_weight = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_weight)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_weight)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(weights, subfolder="unet", use_auth_token=True)

        # 4. The scheduler for denoising
        schedulers = {
            Scheduler.DDIM: DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
            Scheduler.PNDM: PNDMScheduler.from_config(weights, subfolder="scheduler", use_auth_token=True),
            Scheduler.LMSD: LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            ),
        }
        self.scheduler = schedulers[scheduler]

        # put back original logging level
        logging.set_verbosity_warning()

        self.torch_device = torch_device
        if self.torch_device is None:
            if torch.cuda.is_available():
                self.torch_device = "cuda"
            elif torch.has_mps:
                self.torch_device = "mps"
            else:
                self.torch_device = "cpu"
                
        self.vae = self.vae.to(self.torch_device)
        self.text_encoder = self.text_encoder.to(self.torch_device)
        self.unet = self.unet.to(self.torch_device)

    @torch.no_grad()
    def generate_image(
        self,
        prompts: Union[str, List[str]],
        guiding_prompt: str = "",
        init_image: Union[None, str, Image.Image, torch.FloatTensor] = None,
        init_mask: Union[None, str, Image.Image, torch.FloatTensor] = None,
        init_strength: float = 0.8,
        height: int = 512,
        width: int = 512,
        num_steps: int = 25,
        guidance_scale: int = 7.5,
        seed: int = 33,
    ) -> List[Image.Image]:
        """
        Generate images from a list of prompts


        :param prompts: List of prompts
        :param guiding_prompt: String guiding the denoising
        :param init_image: Image to initialize noise of diffuser (disabled if None)
        :param init_mask: Mask to define in painting (disabled if None)
        :param init_strength: Ratio of steps to dedicate to noising initialization image (only if init_image != None)
        :param height: Output height (one of the 2 dimensions need to be 512, needs to be multiple of 8)
        :param width: Output width (one of the 2 dimensions need to be 512, needs to be multiple of 8)
        :param num_steps: Number of steps in the generation process (more steps = more details)
        :param guidance_scale: Float defining the weight of the prompt in the image generation
        :param seed: Seed of the starting noise
        """

        if type(prompts) == str:
            prompts = [prompts]
        elif type(prompts) != list:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompts)}")

        if init_image is not None and type(self.scheduler) == LMSDiscreteScheduler:
            raise ValueError("The LMSD scheduler is not supported for image 2 image. Use another one")
        if 1 < init_strength < 0:
            raise ValueError(f"The value of init_strength should in [0.0, 1.0] but is {init_strength}")

        generator = torch.manual_seed(seed)  # Seed generator to create the inital latent noise
        batch_size = len(prompts)

        # Tokenize the prompts and encode it to get the embeddings
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]
        max_length = text_input.input_ids.shape[-1]

        # Tokenize an empty prompt and encode it to get the embeddings
        uncond_input = self.tokenizer(
            [guiding_prompt] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]

        # Embeddings to condition the U-Net is the concat of both
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Initialize scheduler
        self.scheduler.set_timesteps(num_steps)

        # Generate initial noise
        noise = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        init_timestep = num_steps
        init_latents_orig = None
        offset = 0
        if init_image is None:
            init_latents = noise.to(self.torch_device)
            if hasattr(self.scheduler, "sigmas"):
                init_latents *= self.scheduler.sigmas[0]
        else:
            if isinstance(init_image, str):
                init_image = self.open_image(init_image)

            if isinstance(init_image, Image.Image):
                init_image = self.image_to_tensor(init_image, h=height, w=width)

            # encode the init image into latents and scale the latents
            init_latents = self.vae.encode(init_image.to(self.torch_device)).latent_dist.sample(generator=generator)
            init_latents *= 0.18215

            # prepare init_latents noise to latents
            init_latents = torch.cat([init_latents] * batch_size)
            init_latents_orig = init_latents.to(self.torch_device)

            # get the original timestep using init_timestep
            offset = 1
            init_timestep = int(num_steps * init_strength) + offset
            init_timestep = min(init_timestep, num_steps)
            timesteps = self.scheduler.timesteps[-init_timestep].type(torch.long)
            timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long)

            # add noise to latents using the timesteps
            init_latents = self.scheduler.add_noise(init_latents.cpu(), noise, timesteps).to(self.torch_device)

            # prepare mask
            if init_mask is not None:
                if isinstance(init_mask, str):
                    init_mask = self.open_image(init_mask)

                if isinstance(init_mask, Image.Image):
                    init_mask = self.mask_to_tensor(init_mask, h=height, w=width)

                init_mask = torch.cat([init_mask.to(self.torch_device)] * batch_size)

                # check sizes
                if not init_mask.shape == init_latents.shape:
                    raise ValueError("The mask and init_image should be the same size!")

            noise = noise.to(self.torch_device)

        latents = init_latents

        # Iterative denoising loop
        t_start = max(num_steps - init_timestep + offset, 0)
        for i, t in tqdm(
            enumerate(self.scheduler.timesteps[t_start:]),
            f"Generation {seed}",
            leave=True,
            total=num_steps - t_start,
        ):
            # TODO: specific to MPS, if cuda, autcast should be used around the loop
            if type(self.scheduler) == LMSDiscreteScheduler:
                t = t.type(torch.float32)
            elif type(self.scheduler) == DDIMScheduler:
                t = t.type(torch.long)

            # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            if hasattr(self.scheduler, "sigmas"):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # Predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance (as the latent has both guided and not, we split them here)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            step = t
            if hasattr(self.scheduler, "sigmas"):
                step = i + t_start
            latents = self.scheduler.step(noise_pred, step, latents).prev_sample

            # masking
            if init_mask is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, step)
                # init_latents_proper = init_latents_orig
                latents = (init_latents_proper * init_mask) + (latents * (1 - init_mask))

        return self.latent_to_image_list(latents)

    @torch.no_grad()
    def latent_to_image_list(self, latents):
        """Scale and decode the image latents with vae"""
        latents /= 0.18215

        # Decode latent
        image = self.vae.decode(latents).sample

        return self.tensor_to_image_list(image)

    @staticmethod
    def tensor_to_image_list(tensor):
        """Batch of tensors to PIL image list"""
        image = (tensor / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in images]

    @staticmethod
    def image_to_tensor(image: Image.Image, w: int = None, h: int = None) -> torch.Tensor:
        """Convert a PIL Image to torch tensor"""
        if w is None or h is None:
            w, h = image.size
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    @staticmethod
    def mask_to_tensor(mask: Image.Image, w: int = None, h: int = None) -> torch.Tensor:
        mask = mask.convert("L")
        if w is None or h is None:
            w, h = mask.size
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        mask = mask.resize((w // 8, h // 8), resample=Image.LANCZOS)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)  # weird way to expand dim for batch
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask)
        return mask

    @staticmethod
    def open_image(image_path: str) -> Image.Image:
        return Image.open(image_path)
