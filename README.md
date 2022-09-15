# serving-diffusion

This repo creates a serving containers with all model weights for the following:
- Stable diffusion : txt2img, im2img - [README](./stable-diffusion/README.md)
- RealESRGAN : upscaling, face enhancement - [README](./stable-diffusion/README.md)

Web framework is [fastapi](https://fastapi.tiangolo.com/). 

There are also deployment scripts in each folder to deploy in Google Cloud Platform's Vertex AI Endpoints.

**This repo is still a work in progress. If you find any issues in the code or documentation please file an issue**