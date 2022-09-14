# serving real-esrgan

This repo creates a serving container for Real-ESRGAN for upscaling and face enhancement

# Setup

1. Clone repo
1. `cd` into `real-esrgan` folder.
1. Build container

    ```bash
    docker build . -t gcr.io/{project_id}/real-esrgan:latest
    ```

1. Push the image

    ```bash
    docker push gcr.io/{project_id}/real-esrgan:latest
    ```

1. Run container locally. Needs GPU.

    ```bash
    docker run -p 80:8080 --gpus all -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/{project_id}/real-esrgan:latest -d
    ```