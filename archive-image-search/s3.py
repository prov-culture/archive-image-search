import boto3
from botocore.client import Config
from botocore.exceptions import ClientError  # <- ajout nécessaire
import streamlit as st
from pathlib import Path
from io import BytesIO
from utils import get_logger
logger = get_logger(__name__)
from PIL import Image
from pprint import pprint


class S3():
    def __init__(self) -> None:
        self.client = boto3.client(
            's3',
            aws_access_key_id=st.secrets['OVH']['ACCESS_KEY_ID'],
            aws_secret_access_key=st.secrets['OVH']['SECRET_ACCESS_KEY'],
            endpoint_url=st.secrets['OVH']['ENDPOINT'],
            config=Config(signature_version='s3v4')  # works well with OVH
        )
        self.bucket = 'images-mae'
    
    def upload_files(self, filespath: list[Path]) -> None:
        n = len(filespath)
        text = 'Vérification des photos, merci de patienter...'
        bar = st.progress(0.0, text)
        for i, filepath in enumerate(filespath):
            self.safe_upload_file(filepath)
            bar.progress(value=float((i+1)/n), text=f'{text} ({i}/{n})')
        bar.empty()

    
    def safe_upload_file(self, filepath: Path) -> None:
        try:
            if not self.file_exists(filepath.name):
                self.client.upload_file(filepath, self.bucket, filepath.name)
                logger.info(f'✅ {filepath.name} uploaded')
            else:
                logger.info(f'⏭️ {filepath.name} already on bucket, skipping...')
        except Exception as e:
            logger.error(f'❌ Error while uploading file \"{filepath.name}\": {e}', exc_info=True)
    
    def download_file(self, filename, embeddings=True) -> BytesIO:
        try:
            file_obj = BytesIO()
            self.client.download_fileobj(self.bucket, filename, file_obj)
            file_obj.seek(0)
            logger.info(f'⬇️ {filename} accessed')
            return file_obj
        except Exception as e:
            logger.error(f'❌ Error while retrieving file \"{filename}\": {e}', exc_info=True)
            if not embeddings:
                error_img_path = Path(__file__).parent / 'media/404.png'
                with Image.open(error_img_path) as img:
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")  # ou "PNG", selon ton image
                    buffer.seek(0)  # très important !
                    return buffer
    
    def file_exists(self, filename):
        try:
            self.client.head_object(Bucket=self.bucket, Key=filename)
            return True
        except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
                else:
                    raise
    
    def list_buckets(self):
        return [bucket['Name'] for bucket in self.client.list_buckets()]
    
    def get_all_files(self):
        paginator = self.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket)
        
        all_files = []
        for page in page_iterator:
            if 'Contents' in page:
                all_files.extend(obj['Key'] for obj in page['Contents'])
        return all_files