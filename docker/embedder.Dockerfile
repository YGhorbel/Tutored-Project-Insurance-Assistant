FROM bitnami/spark:3.5.0

USER root

WORKDIR /app

# Install Python packages
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    sentence-transformers==2.2.2 \
    torch==2.1.0 \
    PyYAML==6.0.1 \
    boto3==1.34.0 \
    minio==7.2.0 \
    kafka-python==2.0.2 \
    loguru==0.7.2

# Copy application code
COPY spark /app/spark
COPY utils /app/utils
COPY config /app/config
COPY workers /app/workers

# Download AWS SDK for S3 support
RUN curl -o /opt/bitnami/spark/jars/aws-java-sdk-bundle-1.12.262.jar \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar && \
    curl -o /opt/bitnami/spark/jars/hadoop-aws-3.3.4.jar \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar

ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

USER 1001

CMD ["/opt/bitnami/spark/bin/spark-submit", \
     "--master", "spark://spark-master:7077", \
     "--executor-memory", "4g", \
     "--executor-cores", "2", \
     "/app/spark/embedding_job.py"]