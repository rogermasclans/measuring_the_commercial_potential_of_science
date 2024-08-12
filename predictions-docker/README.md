# Commercial Potential of Science -- Predicitons via Docker


This repository contains the code for the predictions container.
This code is for a [Docker](https://www.docker.com/) container.

## Arguments

The container takes the following arguments that can be passed when creating the model registry in Vertex AI:

- First argument: The name of the .bin model to use for the predictions. This file must be stored in Google Cloud Buckets at (/scicompot/models/). For example, `model.bin`.

## Upload to Google Cloud's Artifact Registry

If you don't have the Google Cloud SDK installed, follow the instructions:

1. Download the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).
2. Install the SDK.
3. Initialize and login using `gcloud init`.
4. Configure docker with `gcloud auth configure-docker us-east1-docker.pkg.dev` (or whatever region you want to use).

You need to have Docker installed and running to create the image and upload it to Google Cloud's Artifact Registry.
To upload the container to Google Cloud's Artifact Registry, run the following:

1. `docker build -t scicompot-predictions:1.0 .`
2. `docker tag scicompot-predictions:1.0 us-east1-docker.pkg.dev/cscicompot/scicompot-trainer/scicompot-predictions:1.0`
3. `docker push us-east1-docker.pkg.dev/com-sci-2/scicompot-trainer/scicompot-predictions:1.0`

Once in Google Cloud's Artifact Registry, you can deploy the container to Google Vertex AI to make predictions.

## Online Predictions

To make online predictions, you need to create a model registry in Google Vertex AI and then deploy an endpoint.
Then you can use the following code in a nodejs application to make predictions:

```js

const axios = require('axios');
const { GoogleAuth } = require('google-auth-library');

const PROJECT_ID = 'scicompot';
const REGION = 'us-east1';
const ENDPOINT_ID = 'xxxx'; ## Substitute with your ENDPOINT

async function makePrediction(predictionText) {
    const keyFile = require('./xxxxx.json'); ## Substitute with your KeyFile
    const auth = new GoogleAuth({
        credentials: keyFile,
        scopes: ['https://www.googleapis.com/auth/cloud-platform'],
    });
    const token = await auth.getAccessToken();

    const response = await axios.post(
        `https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict`,
        {
            instances: [
                predictionText
            ],
        },
        {
            headers: {
                'Authorization': `Bearer ${token}`,
            }
        }
    );
}

makePrediction("Lorem ipsum dolor sit amet.");

```
