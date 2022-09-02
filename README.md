# Stable Diffusion
Code to run Stable Diffusion model from text or text &amp; image or text, image &amp; mask

## Prerequisites
1. You need an Huggingface account to be able to download the weights: https://huggingface.co
1. Locally, you need to create a virtual env
1. You need to install the requirements by running: `pip install -r requirements.txt`
1. On Apple Silicon macs, update to the latest torch version by running:  
`pip3 install --upgrade --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

## How to use it
Please refer to the example notebooks to demonstrate the different functionalities (in order):
- **1_text2image**: From a text (called prompt), let's generate an corresponding image
- **2_image2image**: From a text and an inititalization image, let's generate an image
- **3_in-painting** *aka masked_image2image*: From a text, an inititalization image and a mask, let's patch an image
