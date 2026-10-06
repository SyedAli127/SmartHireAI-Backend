import boto3
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("AWS_S3_BUCKET", "smarthireai-files")
REGION = os.getenv("AWS_REGION", "ap-south-1")

_client = None


def _s3():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=REGION)
    return _client


def public_url(s3_key: str) -> str:
    return f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{s3_key}"


def upload_bytes(data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> str:
    _s3().put_object(Bucket=BUCKET, Key=s3_key, Body=data, ContentType=content_type)
    return public_url(s3_key)


def upload_text(text: str, s3_key: str) -> str:
    return upload_bytes(text.encode("utf-8"), s3_key, "text/plain; charset=utf-8")


def upload_fileobj(fileobj, s3_key: str, content_type: str = "application/octet-stream") -> str:
    _s3().upload_fileobj(fileobj, BUCKET, s3_key, ExtraArgs={"ContentType": content_type})
    return public_url(s3_key)
