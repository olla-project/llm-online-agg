# LLM Online Aggregation (OLLA)

## Abstract

This repository contains the implementation of the LLM Online Aggregation (OLLA) system.

## System Architecture

OLLA consists of four main components:

1. **Proxy Server** - Receives queries and distributes them via Kafka
2. **PySpark Client** - Submits queries and consumes streaming results
3. **LLM Service** - Processes natural language queries
4. **Kafka** - Message queue for data transport

## Prerequisites

- Apache Spark
- Apache Kafka service running
- Python
- LLM service accessible via vLLM API

## Usage

### 1. Configure Proxy Server

Edit the `proxy_server/.env` configuration file with your Kafka and LLM settings:

```
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
LLM_BASE_URL=http://0.0.0.0:9999/v1
LLM_MODEL=/path/to/your/model
```

### 2. Launch Proxy Server

```bash
cd proxy_server
python proxy_server.py
```

### 3. Run PySpark Client

```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 \
  --master local[*] \
  pyspark_client.py \
  --query <query_module> \
  --proxy-url http://localhost:9981/query
```

The client accepts these parameters:
- `--query` (required): Query module name (e.g., `exp_arxiv_groupby_base`)
- `--proxy-url` (optional): Proxy Server URL

## Query Modules

example query modules:

- `exp_bbc_filter`
- `exp_document_avg`
- `exp_review_category_dynamic`