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
