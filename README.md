# serving-diffusion - WIP

This repo creates a serving container for stable diffusion.

**Dockerfile is messy and needs to be cleaned up**

# Setup

1. Clone repo
1. Build container

  ```bash
  docker build . -t gcr.io/{project_id}/stable-diffusion:latest
  ```
1. Push the image

    ```bash
    docker push gcr.io/{project_id}/stable-diffusion:latest
    ```

1. Run container. Needs GPUs.

  ```bash
  docker run -p 80:8080 --gpus all -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict diffusion -d
  ```

1. Make a prediction

  ```bash
  curl -X POST -d @request.json -H "Content-Type: application/json; charset=utf-8" localhost/predict >> response.json
  ```
  
 1. The result is a json containing a base64 encoded image of the grid samples, not the actual images.
