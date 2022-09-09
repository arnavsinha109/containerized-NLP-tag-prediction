from concurrent.futures import process
import boto3
import pandas as pd
import spacy
import sagemaker
import json


def get_file_from_s3():
    '''
    define S3 bucket, file source(in S3 bucket), file destination(in container)
    '''
    print("initiate s3 connection")
    s3 = boto3.client('s3')
    print("set container destination")
    spn_container_path_file_name = "data/data_for_prediction.xlsx"
    print("set source s3 bucket")
    str_bucket_name = "sagemaker-us-west-2-xxxxxxxxxxxx"   # Dummy AWS S3 bucket name
    print("set source file path")
    spn_s3_path_file_name = "fasttext/prediction_data/data_for_prediction_on_AWS.xlsx"
    print("download file from source to destination")
    s3.download_file(str_bucket_name, spn_s3_path_file_name, spn_container_path_file_name)

def read_data_for_prediction():
    '''
    read data downloaded from S3 into pandas dataframe
    '''
    data_for_prediction = pd.read_excel(r"data/data_for_prediction.xlsx", sheet_name="Sheet1")
    return data_for_prediction

def spacy_text_cleaning():
    '''
    creating spacy nlp object to process data into required format
    object created using pretrained spacy pipeline inside the container
    '''
    global data_for_prediction
    nlp = spacy.load("en_core_web_sm")

    # column with text data
    col_name_to_tokenize = 'Description'
    # column to be created with cleaned text data
    processed_col_name = 'processed_desc'
    # tokenising the file content
    input_text_pd_tokenized = [nlp(str(item.lower())) for item in data_for_prediction[col_name_to_tokenize]]
    # cleaning the file
    cleaned_docs = list()
    # utilizing the pre defined en_core_web_sm spaCy pipeline to do basic cleaning - remove whitespace, numbers, single letter words, stop words and pronouns
    for spacy_doc in input_text_pd_tokenized:
        cleaned_spacy_doc = []
        for token in spacy_doc:
            if token.is_space:
                continue
            if token.like_num:
                continue
            if len(token)<2:
                continue
            if token.is_stop:
                continue
            if token.lemma_ == "-PRON-":
                continue
            cleaned_spacy_doc.append(token.text)
        cleaned_docs.append(" ".join(cleaned_spacy_doc))

    # using the cleaned_docs list object to create a pandas series
    cleaned_docs_pd_series = pd.Series(cleaned_docs, name=processed_col_name)

    # merging the pandas series with the original data
    processed_data_for_prediction = pd.concat([data_for_prediction,cleaned_docs_pd_series], axis=1)
    return processed_data_for_prediction

def create_payload(processed_data):
    '''
    blazing text invocations requires payload json in the following format:
    { "instances": [sentence1, sentence2, sentence3....]}
    '''
    data_as_list = processed_data["processed_desc"].tolist()
    payload = {"instances": data_as_list}
    return payload

def call_endpoint_for_inference(payload):
    '''
    invoke blazingtext endpoint
    note- payload needs to be passed as a JSON in the invocation
    '''
    runtime= boto3.client('runtime.sagemaker')
    
    # Endpoint name is provided when the model has been deployed in Amazon SageMaker
    # Given endpoint name is a dummy 
    # Follow link in repo README to understand steps of endpoint deployment for a FastText model
    result = runtime.invoke_endpoint(EndpointName='blazingtext-2022-08-16-xx-yy-zz-aaa',
    Body=json.dumps(payload),
    ContentType='application/json')

    return result

def process_labels(inference_results_body):
    '''
    process labels and probabilities in pandas.Series object
    merge the series with the original data frame
    '''
    global data_for_prediction

    labels = list()
    probs = list()
    for item in inference_results_body:
        labels.append(item["label"][0])
        probs.append(item["prob"][0])

    data_with_prediction = pd.concat([data_for_prediction,pd.Series(labels,name="label"),pd.Series(probs,name="prob")], axis=1)
    return data_with_prediction

def clean_predicted_data():
    '''
    clean label predictions
    '''
    global data_with_prediction
    print(type(data_with_prediction))

    data_with_prediction["label"] = data_with_prediction["label"].replace('__label__','',regex=True)
    data_with_prediction["label"] = data_with_prediction["label"].replace('_',' ',regex=True)
    data_with_prediction["label"] = data_with_prediction["label"].str.title()
    
def write_data_to_excel():
    '''
    save data with predicted labels inside container
    '''
    global data_with_prediction
    data_with_prediction.to_excel(r"data/data_with_prediction.xlsx", sheet_name="Sheet1", encoding='utf-8', index=False)



def put_data_into_s3():
    '''
    define s3 bucket, file source(inside container), file destination(in S3)
    '''    
    s3 = boto3.client('s3')
    spn_local_path_file_name = "data/data_with_prediction.xlsx"
    str_bucket_name = "sagemaker-us-west-2-xxxxxxxxxxxx"
    spn_remote_path_file_name = "fasttext/predicted_data/data_with_prediction_on_AWS.xlsx"
    s3.upload_file(spn_local_path_file_name, str_bucket_name, spn_remote_path_file_name)
    


if __name__=="__main__":
    global data_for_prediction
    global data_with_prediction
    # download from s3
    get_file_from_s3()

    print("load data from destination in pandas")
    data_for_prediction = read_data_for_prediction()

    print(data_for_prediction.head())

    print("process data")
    processed_data_for_prediction = spacy_text_cleaning()
    print(processed_data_for_prediction.head())

    # create payload for inference
    payload_for_inference = create_payload(processed_data_for_prediction)

    # pass payload to the endpoint
    inference = call_endpoint_for_inference(payload = payload_for_inference)
    
    # extract labels from the inference
    inference_results = json.loads(inference['Body'].read())

    # add labels to the predicted data
    data_with_prediction = process_labels(inference_results_body=inference_results)
    print(data_with_prediction.head())

    # clean the labels
    clean_predicted_data()

    # save data with predictions to excel
    write_data_to_excel()

    # transfer file from container to s3
    put_data_into_s3()