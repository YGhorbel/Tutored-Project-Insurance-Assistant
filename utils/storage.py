"""S3/MinIO storage abstraction layer"""
import io
import json
from typing import Any, Dict, List, Optional
from minio import Minio
from minio.error import S3Error
from loguru import logger
import pickle

class StorageClient:
    """Unified storage client for MinIO/S3"""
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        """Initialize storage client"""
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        logger.info(f"Storage client initialized: {endpoint}")
    
    def ensure_bucket(self, bucket_name: str):
        """Create bucket if it doesn't exist"""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
            else:
                logger.debug(f"Bucket exists: {bucket_name}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket {bucket_name}: {e}")
            raise
    
    def upload_file(self, bucket: str, object_name: str, file_path: str):
        """Upload file to storage"""
        try:
            self.ensure_bucket(bucket)
            self.client.fput_object(bucket, object_name, file_path)
            logger.info(f"Uploaded {file_path} to {bucket}/{object_name}")
        except S3Error as e:
            logger.error(f"Upload error: {e}")
            raise
    
    def upload_bytes(self, bucket: str, object_name: str, data: bytes):
        """Upload bytes to storage"""
        try:
            self.ensure_bucket(bucket)
            data_stream = io.BytesIO(data)
            self.client.put_object(
                bucket, object_name, data_stream, length=len(data)
            )
            logger.info(f"Uploaded bytes to {bucket}/{object_name}")
        except S3Error as e:
            logger.error(f"Upload error: {e}")
            raise
    
    def upload_json(self, bucket: str, object_name: str, data: Dict):
        """Upload JSON data to storage"""
        json_bytes = json.dumps(data).encode('utf-8')
        self.upload_bytes(bucket, object_name, json_bytes)
    
    def download_file(self, bucket: str, object_name: str, file_path: str):
        """Download file from storage"""
        try:
            self.client.fget_object(bucket, object_name, file_path)
            logger.info(f"Downloaded {bucket}/{object_name} to {file_path}")
        except S3Error as e:
            logger.error(f"Download error: {e}")
            raise
    
    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        """Download object as bytes"""
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            logger.debug(f"Downloaded {bucket}/{object_name}")
            return data
        except S3Error as e:
            logger.error(f"Download error: {e}")
            raise
    
    def download_json(self, bucket: str, object_name: str) -> Dict:
        """Download JSON object"""
        data = self.download_bytes(bucket, object_name)
        return json.loads(data.decode('utf-8'))
    
    def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """List objects in bucket with prefix"""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"List error: {e}")
            return []
    
    def delete_object(self, bucket: str, object_name: str):
        """Delete object from storage"""
        try:
            self.client.remove_object(bucket, object_name)
            logger.info(f"Deleted {bucket}/{object_name}")
        except S3Error as e:
            logger.error(f"Delete error: {e}")
            raise