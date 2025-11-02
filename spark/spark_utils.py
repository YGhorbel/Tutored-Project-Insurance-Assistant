"""Utilities for Spark jobs"""
from typing import Dict, Any
import json


class SparkS3Manager:
    """Helper for S3 operations in Spark"""
    
    @staticmethod
    def configure_s3(spark_session, endpoint: str, access_key: str, secret_key: str):
        """Configure Spark for S3 access"""
        hadoop_conf = spark_session._jsc.hadoopConfiguration()
        
        hadoop_conf.set("fs.s3a.endpoint", endpoint)
        hadoop_conf.set("fs.s3a.access.key", access_key)
        hadoop_conf.set("fs.s3a.secret.key", secret_key)
        hadoop_conf.set("fs.s3a.path.style.access", "true")
        hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")
        hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        hadoop_conf.set("fs.s3a.aws.credentials.provider", 
                       "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    
    @staticmethod
    def s3_path(bucket: str, key: str = "") -> str:
        """Generate S3 path"""
        return f"s3a://{bucket}/{key}" if key else f"s3a://{bucket}"


def batch_iterator(iterable, batch_size: int):
    """Yield successive batches from iterable"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch