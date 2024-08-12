# Commercial Potential of Science -- Trainer via Docker

This repository contains the code for the trainer container.
This code is for a [Docker](https://www.docker.com/) container.

## Arguments

The container takes the following arguments, which can be passed when creating the model registry in Vertex AI:

- First argument: The name of the file with the data to train the model. The file must be in Google Cloud Buckets (/scicompot/training-data/). For example, `training-data.csv`.
- Second argument: The name of the final .bin model. The file will be stored in Google Cloud Buckets at (/scicompot/models/). For example, `model.bin`.

## Upload to Google Cloud's Artifact Registry

If you don't have the Google Cloud SDK installed, follow the instructions:

1. Download the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).
2. Install the SDK.
3. Initialize and login using `gcloud init`.
4. Configure docker with `gcloud auth configure-docker us-east1-docker.pkg.dev` (or whatever region you want to use).

You need to have Docker installed and running to create the image and upload it to Google Cloud's Artifact Registry.
To upload the container to Google Cloud's Artifact Registry, run the following:

1. `docker build -t {trainer_name:version} .`
2. `docker tag {trainer_name:version} {gc_region}-docker.pkg.dev/{gc_project}/{gc_artifact_repo_name}/{trainer_name:version}`
3. `docker push {gc_region}-docker.pkg.dev/{gc_project}/{gc_artifact_repo_name}/{trainer_name:version}`

Example:

1. `docker build -t scicompot-trainer:1.0 .`
2. `docker tag scicompot-trainer:1.0 us-east1-docker.pkg.dev/scicompot/scicompot-trainer/scicompot-trainer:1.0`
3. `docker push us-east1-docker.pkg.dev/scicompot/scicompot-trainer/scicompot-trainer:1.0`

Once in Google Cloud's Artifact Registry, you can deploy the container to Google Vertex AI to start the training.

## Default arguments

1. dynamic_data = "training_data_small_test.csv"
2. local_model_name = 'small_test_model.bin'
3. bucket_name = 'scicompot'
4. epochs = 5
5. drop_out_rate = 0.3
6. lr = 2e-5 
7. model_used = 'scibert'
8. training_data_folder = 'training-data'
9. batch_size = 16
