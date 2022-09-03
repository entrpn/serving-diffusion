import argparse
from google.cloud import aiplatform

def deploy_model_with_dedicated_resources_sample(
    model,
    machine_type: str,
    endpoint = None,
    deployed_model_display_name = None,
    traffic_percentage = 0,
    traffic_split= None,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    accelerator_type = None,
    accelerator_count = None,
    explanation_metadata = None,
    explanation_parameters = None,
    metadata = (),
    sync: bool = True,
):
    # The explanation_metadata and explanation_parameters should only be
    # provided for a custom trained model and not an AutoML model.
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        traffic_percentage=traffic_percentage,
        traffic_split=traffic_split,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        metadata=metadata,
        sync=sync,
    )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-id",
        type=str,
        help="the gcp project id"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="name of gcp bucket"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="gcp region"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="stable-difussion",
        help="name of model"        
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        help="name of image in gcr. Ex: gcr.io/project-name/stable-diffusion"
    )

    opt = parser.parse_args()
    
    aiplatform.init(project=opt.project_id,location=opt.region)
    
    model = aiplatform.Model.upload(
    display_name=opt.model_name,
    serving_container_image_uri=opt.image_uri,
    serving_container_ports=[8080],
    serving_container_predict_route="/predict"
    )
    
    model = deploy_model_with_dedicated_resources_sample(model,
                                                        "n1-standard-8",
                                                        traffic_percentage=100,
                                                        accelerator_type='NVIDIA_TESLA_T4',
                                                        accelerator_count=1)




if __name__ == "__main__":
    main()