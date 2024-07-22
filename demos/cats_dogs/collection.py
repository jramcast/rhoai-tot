import os
from typing import Union
import boto3
import pandas as pd
from pathlib import Path


def s3_download_file(s3_filepath: str, destination: Union[str, os.PathLike]):
    """
    Download a file from a S3 bucket.

    s3_filepath must be file path relative to the bucket
    """
    key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint = os.getenv("AWS_S3_ENDPOINT")
    bucket_name = os.getenv("AWS_S3_BUCKET")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        use_ssl=True,
    )

    # Create dir to cache downloaded data
    data_dir = Path(destination).parent
    if not data_dir.is_dir():
        os.makedirs(data_dir)

    s3.download_file(
        bucket_name, s3_filepath, destination
    )


def load():
    S3_TRAIN_FILE = os.getenv("S3_TRAIN_FILE")

    data_file = ".cache/train.csv"

    s3_download_file(S3_TRAIN_FILE, destination=data_file)

    data = pd.read_csv(data_file)

    return data


if __name__ == "__main__":
    load()
