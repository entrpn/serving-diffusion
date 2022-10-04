# serving codeformer
Creates a serving container for the repo [CodeFormer](https://github.com/sczhou/CodeFormer)

![](../images/01.jpeg) 
![](../images/02.png)
## Setup

1. Clone repo if you haven't. Navigate to the `codeformer` folder.
1. Build the container. Don't forget to change the `project_id` to yours.

    ```bash
    docker build . -t gcr.io/{project_id}/codeformer:latest
    ```

1. Run container. You need [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) and a GPU.

    ```bash
    docker run -p 80:8080 --gpus all -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/{project_id}/codeformer:latest -d
    ```

1. Make a prediction

    ```bash
    python generate_requeset.py
    curl -X POST -d @request.json -H "Content-Type: application/json; charset=utf-8" localhost/predict > response.json
    ```

## Deploy in Vertex AI

You'll need to enable Vertex AI and have authenticated with a service account that has the Vertex AI admin or editor role.

1. Push the image

    ```bash
    gcloud auth configure-docker
    docker build . -t gcr.io/{project_id}/codeformer:latest
    gcloud push gcr.io/{project_id}/codeformer:latest
    ```

1. Deploy in Vertex AI endpoints.

    ```bash
    python gcp_deploy.py --image-uri gcr.io/<project_id>/codeformer:latest
    ```

1. Test the endpoint.

    ```bash
    python generate_request_vertex.py
    ```