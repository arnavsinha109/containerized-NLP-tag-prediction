# containerized-NLP-text-processing-classification
Docker container codes for NLP preprocessing using spaCy and prediction using FastText model deployed as a BlazingText endpoint (Amazon SageMaker)

## Process Flow
1. Connect to S3 bucket
2. Extract data from S3
3. Process the text data using spaCy pipeline
4. Predict tags by invoking BlazingText endpoint (deployed using Amazon SageMaker)
5. Store predictions in S3

## Setting up the container
1. Build image - ```docker build --tag image-name . ```
2. Running container on local system- ``` docker run image-name -e AWS_ACCESS_KEY_ID=value -e AWS_SECRET_ACCESS_KEY=value -e AWS_DEFAULT_REGION=value```

## Notes
1. When running the container best practice would be to use credential vaults to pass the above variables for successful execution. AWS Secrets Manager would do the job when deploying the container on AWS
2. The text classification model was built using [FastText](https://fasttext.cc/) in Python. The steps to train and save a FastText text classification model are given here given here - [Supervised Text Classification Model Building](https://fasttext.cc/docs/en/supervised-tutorial.html)
3. Steps to deploy FastText model as an endpoint - [Model Deployment Steps](https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/blazingtext_hosting_pretrained_fasttext/blazingtext_hosting_pretrained_fasttext.html)
4. Steps to deploy container using Amazon ECS and Fargate - [Container Deployment Steps](https://towardsdatascience.com/deploying-a-docker-container-with-ecs-and-fargate-7b0cbc9cd608)
5. The trained spaCy pipeline in the pre-compile-lib-dependency folder can be found here - [spaCy Pipeline Link](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.4.0)
