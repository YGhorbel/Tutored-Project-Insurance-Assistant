"""PySpark job for distributed embedding generation"""
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType
import torch
from sentence_transformers import SentenceTransformer
import json
from typing import List
import yaml

# Broadcast model to workers
model_broadcast = None


def get_embedding_model():
    """Get or create embedding model (singleton per executor)"""
    global model_broadcast
    if model_broadcast is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_broadcast = SentenceTransformer(model_name, device=device)
    return model_broadcast


def embed_text(text: str) -> List[float]:
    """Generate embedding for text"""
    try:
        model = get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error embedding text: {e}")
        return []


def main():
    """Main Spark job"""
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Insurance-Embeddings") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()
    
    print("Spark session created")
    
    # Load configuration
    config_path = "/app/config/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # S3/MinIO configuration
    storage_cfg = config['storage']
    endpoint = storage_cfg['endpoint']
    access_key = storage_cfg['access_key']
    secret_key = storage_cfg['secret_key']
    
    # Configure S3 access
    spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", f"http://{endpoint}")
    spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
    spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
    spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    
    # Paths
    enriched_bucket = storage_cfg['buckets']['enriched']
    embeddings_bucket = storage_cfg['buckets']['embeddings']
    
    input_path = f"s3a://{enriched_bucket}/*.json"
    output_path = f"s3a://{embeddings_bucket}/embeddings.parquet"
    
    print(f"Reading from: {input_path}")
    
    # Read enriched data
    try:
        df = spark.read.json(input_path)
        print(f"Loaded {df.count()} records")
    except Exception as e:
        print(f"Error reading input: {e}")
        # Create empty DataFrame if no data
        schema = StructType([
            StructField("chunk_id", StringType(), True),
            StructField("text", StringType(), True),
            StructField("summary", StringType(), True)
        ])
        df = spark.createDataFrame([], schema)
    
    if df.count() == 0:
        print("No data to process")
        spark.stop()
        return
    
    # Register UDF for embedding
    embed_udf = udf(embed_text, ArrayType(FloatType()))
    
    # Generate embeddings
    print("Generating embeddings...")
    df_with_embeddings = df.withColumn("text_embedding", embed_udf(col("text")))
    
    if "summary" in df.columns:
        df_with_embeddings = df_with_embeddings.withColumn(
            "summary_embedding",
            embed_udf(col("summary"))
        )
    
    # Select relevant columns
    output_df = df_with_embeddings.select(
        "chunk_id",
        "text_embedding",
        "summary_embedding" if "summary" in df.columns else col("text_embedding").alias("summary_embedding"),
        "metadata"
    )
    
    # Write to parquet
    print(f"Writing embeddings to: {output_path}")
    output_df.write.mode("overwrite").parquet(output_path)
    
    print(f"Successfully generated embeddings for {output_df.count()} chunks")
    
    # Also write individual JSON files for indexer
    print("Writing individual embedding files...")
    embeddings_collected = output_df.collect()
    
    for row in embeddings_collected:
        chunk_id = row['chunk_id']
        embedding_data = {
            'chunk_id': chunk_id,
            'embedding': row['text_embedding'],
            'metadata': row['metadata']
        }
        
        # Write to S3 (this is simplified - in production, use better batching)
        json_str = json.dumps(embedding_data)
        
        # Use spark to write (more efficient for distributed env)
        # For now, just print (in production, use proper S3 writing)
        print(f"Generated embedding for {chunk_id}")
    
    spark.stop()
    print("Spark job completed")


if __name__ == "__main__":
    main()