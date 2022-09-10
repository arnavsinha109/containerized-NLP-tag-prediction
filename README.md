# containerized-NLP-text-processing-classification
Docker container codes for NLP preprocessing using spaCy and prediction using FastText model deployed as a BlazingText endpoint (Amazon SageMaker)

## Background
* I built a supervised Text Classification model using [FastText](https://fasttext.cc/) in Python. I wanted to scale and deploy this model on AWS SageMaker to get predictions for new data
* When training the model, I used spaCy pretrained pipelines to clean the text. Similarly, the new data would need to be cleaned before being passed to the model
* This container was designed to replicate the data processing pipeline needed for good predictions

## Challenges encountered in model building
1. The data needed to be cleaned to train a robust model. I used spaCy to perform data cleaning before using it for model training
2. The quality of model predictions depends on multiple factors. My baseline models did not yield good results. Hence, I augmented my model training by using pretrained FastText models available here - [Pretrained FastText Vectors](https://fasttext.cc/docs/en/english-vectors.html). I was able to achieve an **F1 score of 0.96** when I used the pretrained models

## Container Process Flow
1. Connect to S3 bucket
2. Extract data from S3
3. Process the text data using spaCy pipeline
4. Predict tags by invoking BlazingText endpoint (deployed using Amazon SageMaker)
5. Store predictions in S3

## Setting up the container (in CLI)
1. Build image - ```docker build --tag image-name . ```
2. Running container on local system- ``` docker run image-name -e AWS_ACCESS_KEY_ID=value -e AWS_SECRET_ACCESS_KEY=value -e AWS_DEFAULT_REGION=value```

## Notes
1. When running the container best practice would be to use credential vaults to pass the above variables for successful execution. AWS Secrets Manager would do the job when deploying the container on AWS
2. Steps to train and save a FastText text classification model are given here given here - [Supervised Text Classification Model Building](https://fasttext.cc/docs/en/supervised-tutorial.html)
3. Steps to deploy FastText model as an endpoint - [Model Deployment Steps](https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/blazingtext_hosting_pretrained_fasttext/blazingtext_hosting_pretrained_fasttext.html)
4. Steps to deploy container using Amazon ECS and Fargate - [Container Deployment Steps](https://towardsdatascience.com/deploying-a-docker-container-with-ecs-and-fargate-7b0cbc9cd608)
5. The trained spaCy pipeline in the pre-compile-lib-dependencies folder can be found here - [spaCy Pipeline Link](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.4.0)
