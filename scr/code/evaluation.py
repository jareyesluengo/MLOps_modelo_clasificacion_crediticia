import os
import sys
import json
import joblib
import pathlib
import tarfile
import pickle
import logging
import argparse

import boto3
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)
# sm_client = boto3.client("sagemaker")

def get_approved_package(model_package_group_name, sm_client):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return approved_packages[0]
        # return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # bucket variables
    parser.add_argument("--bucket-output", type=str, required=True)
    parser.add_argument("--region-name", type=str, required=True)

    return parser.parse_known_args()


model_path = "/opt/ml/processing/model/model.tar.gz"

if __name__ == "__main__":
    args, _ = parse_args()
    
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("./model")
    
    model = joblib.load("./model/model.joblib")
    validation_dir = "/opt/ml/processing/validation/"
    validation_data = pd.read_csv(os.path.join(validation_dir, "validation.csv"))
    x_val = validation_data.iloc[:, :-1].to_numpy()
    y_val = validation_data.iloc[:, -1].to_numpy()

    logger.info("Performing predictions against test data.")
    y_pred = model.predict(x_val)

    weighted_precision = precision_score(y_val, y_pred, average='weighted')
    weighted_recall = recall_score(y_val, y_pred, average='weighted')
    accuracy = accuracy_score(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)

    logger.debug("accuracy: {}".format(accuracy))
    logger.debug("weighted_precision: {}".format(weighted_precision))
    logger.debug("weighted_recall: {}".format(weighted_recall))
    logger.debug("confusion_matrix: {}".format(conf_matrix))
    
    print(f'accuracy: {accuracy}, weighted precision: {weighted_precision}, weighted recall: {weighted_recall}')

    # metricas de evaluación
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "weighted_precision": {"value": weighted_precision, "standard_deviation": "NaN"},
            "weighted_recall": {"value": weighted_recall, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1]), "2": int(conf_matrix[0][2])},
                "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1]), "2": int(conf_matrix[1][2])},
                "2": {"0": int(conf_matrix[2][0]), "1": int(conf_matrix[2][1]), "2": int(conf_matrix[2][2])},
            },
        },
    }

    # model evaluation
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    # retrieve model metrics     
    sm_client = boto3.client("sagemaker", region_name=args.region_name)
    pck = get_approved_package('CreditScoreModel', sm_client)
    model_description = sm_client.describe_model_package(ModelPackageName=pck["ModelPackageArn"])
    
    last_eval_metrics = model_description['ModelMetrics']['ModelQuality']['Statistics']['S3Uri']
    print(f'S3 path of last eval metrics {last_eval_metrics}')
    
    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(args.bucket_output)

    metrics_dir = "/opt/ml/processing/production"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    xs = last_eval_metrics.split('/')[3:]
    output_eval = os.path.join(metrics_dir, 'evaluation.json')
    print(f'Local path {output_eval}')
    s3_bucket.download_file('/'.join(xs), output_eval)