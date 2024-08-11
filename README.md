Last updated: August 2024

# End-to-end example with the Diaster Tweet dataset

We provide an end-to-end example with the 'Disaster Tweet Classification' example, which classifies a tweet (text string) into whether it is a disaster or not. Previously, we performed an exploratory data analysis and ran model architecture and hyperparameter searches in a separate repo. In this repo, we implement:
1. A training pipeline in `./training_pipeline`
    - Notably we use TfidfVectorizer and logistic regression, which constitute a computationally efficient solution
    - Folder structure:
      - `data`: training dataset: ideally it is downloaded from blob storage
      - `data_train`: transformed training data
      - `src`: Python code for the training pipeline
      - `run_train_pipeline.py`: training pipeline entrypoint
      - `train_config.yml`: training pipeline config
      - `model_assets`: the fitted objects/model together with evaluation scores are saved to this folder, which in a realife scenario they will be uploaded to blob storage
      - `requirements.txt`
    - To run the training pipeline:
       1. Supply training dataset, containing `train.csv`
       2. Run `pip install -r requirements.txt`
       3. Set `train_config.yml`
       4. Run `python3 run_train_pipeline.py -c train_config.yml`
2. A corresponding inference FastAPI backend in `./inference_backend`
   - Folder structure:
     - `model_assets`: contains the fitted objects/model from running the training pipeline; in a realife scenario they will be downloaded from blob storage
     - `src`: Python code for the inference pipeline corresponding to the training pipeline using the fitted `model_assets`
     - `main.py`: FastAPI app
     - `Dockerfile` and `docker_compose.yml`: used to build and start a container
   - To test the backend:
     1. Supply `./model_assets` by running the training pipeline
     2. Run `docker build . -t disaster/inference_backend:dev` to build the Docker image locally
     3. Run `docker compose -f "inference_backend/docker_compose.yml" up -d --build` to spin up the container
     4. Test the endpoint using Swagger UI at `http://localhost:3100/docs`
     5. To kill the container, run: `docker compose -f "inference_backend/docker_compose.yml" down`

