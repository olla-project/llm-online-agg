#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PySpark Client for Online Aggregation

Sends queries to proxy server, consumes results from Kafka, and displays progress.

Usage: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 pyspark_client.py
"""

import os
import time
import logging
import requests
import uuid
import io
import sys
import json
import importlib
import argparse
from contextlib import contextmanager
from functools import partial
from typing import Dict, Any, List, Optional, Union

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, avg, when, lit, count, explode, max
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType,
    IntegerType,
    DoubleType,
    ArrayType,
    FloatType,
    TimestampType,
)
from pydantic import BaseModel
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.WARNING)

# Configuration
# Default values that can be overridden by command-line arguments
PROXY_SERVER_URL = "http://localhost:9981/query"


def send_query_to_proxy(query: str, request_id: str = None) -> Dict[str, Any]:
    """Send query request to proxy server"""
    if not request_id:
        request_id = str(uuid.uuid4())

    # Basic request parameters
    payload = {
        "query": query,
        "request_id": request_id,
        "data_path": DATA_PATH,
    }

    # Load different request parameters according to mode
    payload["vectorization"] = globals().get("vectorization", False)
    payload["sample_config"] = globals().get("sample_config", "none")
    payload["input_optimizaion"] = globals().get("input_optimizaion", False)
    payload["output_optimization"] = globals().get("output_optimizaion", "none")
    payload["fields"] = globals().get("fields", None)
    payload["dynamic_mode"] = globals().get("dynamic_mode", "adjust")

    payload["output_json_schema"] = OutputSchema.model_json_schema()

    try:
        response = requests.post(PROXY_SERVER_URL, json=payload)
        response.raise_for_status()

        response_data = response.json()
        logger.info(f"Query sent with ID: {request_id}")

        if not response_data.get("kafka_topic") or not response_data.get(
            "kafka_bootstrap_servers"
        ):
            raise ValueError("Missing Kafka connection info in response")

        return response_data
    except Exception as e:
        logger.error(f"Error sending query: {e}")
        return {"status": "error", "message": str(e)}


class ProgressTracker:
    """Message counter - used for tracking processing progress"""

    def __init__(self, total):
        self.processed_count = 0
        self.total = total

    def update(self, count):
        self.processed_count += count

    def get_progress(self):
        return self.processed_count, self.total


def setup_kafka_stream(spark, kafka_topic, kafka_bootstrap_servers, schema):
    kafka_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
        .option("subscribe", kafka_topic)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )
    parsed_df = (
        kafka_stream.selectExpr("CAST(value AS STRING) as json_value")
        .select(from_json("json_value", schema).alias("data"))
        .select("data.*")
    )

    return parsed_df


def setup_stratum_stream(spark, kafka_stratum_topic, kafka_bootstrap_servers):
    # Define schema for single element
    element_schema = StructType(
        [
            StructField("class_label", StringType()),
            StructField("proportion", FloatType()),
            StructField("confidence_interval", ArrayType(FloatType())),
            StructField("timestamp", StringType()),
            # StructField("timestamp", TimestampType()),
        ]
    )
    schema = ArrayType(element_schema)
    kafka_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
        .option("subscribe", kafka_stratum_topic)
        .option("startingOffsets", "latest")
        .load()
    )
    parsed_df = (
        kafka_stream.selectExpr("CAST(value AS STRING) as json_value")
        .select(from_json("json_value", schema).alias("data"))
        .select(explode("data").alias("item"))
        .select("item.*")
    )
    return parsed_df


def set_progress_tracker_stream(df, progress_tracker):
    def track_progress(batch_df, batch_id):
        count = batch_df.count()
        if count > 0:
            progress_tracker.update(count)
            processed, total = progress_tracker.get_progress()
            percent = round(processed / total * 100, 1) if total > 0 else 0
            # logger.info(f"Progress: {processed}/{total} ({percent}%)")

    return (
        df.writeStream.outputMode("append")
        .foreachBatch(track_progress)
        .trigger(processingTime="1 seconds")
        .start()
    )


stratum_info: dict[str, dict] = {}


def create_display_function(progress_tracker, query_processor=None):
    assert query_processor, "query_processor must be provided."

    # Global view to store all processed data - using dictionary instead of DataFrame to support updates
    # Note: This approach works for medium-scale data, for very large-scale data more complex state management may be needed
    all_data = []

    def display_results(batch_df, batch_id):
        # Add current batch data to global data
        nonlocal all_data
        batch_rows = batch_df.collect()
        all_data.extend(batch_rows)

        # Write to local file
        # with open(f"logs/hyper-para/filter/bbc-count-k4.json", "w") as f:
        #     for row in all_data:
        #         row_dict = row.asDict()
        #         f.write(json.dumps(row_dict) + "\n")

        buf = io.StringIO()

        # 捕获DataFrame.show()的输出
        @contextmanager
        def capture_show():
            old_stdout = sys.stdout
            sys.stdout = temp_stdout = io.StringIO()
            try:
                yield temp_stdout
            finally:
                sys.stdout = old_stdout

        # Display title
        buf.write(f"=== Query Results (Batch ID: {batch_id}) ===\n")

        # Convert complete data back to DataFrame for aggregation processing
        if all_data:
            # Create new DataFrame from accumulated data
            complete_df = batch_df.sparkSession.createDataFrame(
                all_data, batch_df.schema
            )
            # Apply user's query processor
            result_df = query_processor(complete_df)

            result_row = result_df.collect()
            timestamp = str(datetime.now())
            # with open(f"logs/base/arxiv_stratum_info.json", "a") as f:
            #     for row in result_row:
            #         row_dict = row.asDict()
            #         row_dict["timestamp"] = timestamp
            #         f.write(json.dumps(row_dict) + "\n")

            # Display query results
            with capture_show() as show_out:
                result_df.show(truncate=False)
            buf.write(show_out.getvalue())
        else:
            buf.write("\nNo data\n")

        def create_progress_bar(processed, total, bar_len=50):
            """Create progress bar display"""
            percent = (processed / total * 100) if total > 0 else 0
            filled = int(bar_len * processed // total) if total > 0 else 0
            bar = "█" * filled + "-" * (bar_len - filled)
            return bar, percent

        # Get progress information and display progress bar
        processed, total = progress_tracker.get_progress()
        bar, percent = create_progress_bar(processed, total)

        buf.write(f"Progress: |{bar}| {percent:.1f}% Complete\n")
        buf.write(f"Processed: {processed}/{total} files\n\n")
        buf.write(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        buf.write("=" * 60)

        # Clear screen and print
        os.system("clear" if os.name == "posix" else "cls")
        print(buf.getvalue())

    return display_results


def start_kafka_consumer(
    spark,
    kafka_topic: str,
    kafka_stratum_topic: str,
    kafka_bootstrap_servers: str,
    total_data_size: int,
):
    progress = ProgressTracker(total_data_size)

    # Set up Kafka stream and progress tracking stream
    parsed_df = setup_kafka_stream(
        spark, kafka_topic, kafka_bootstrap_servers, PYSPARK_SCHEMA
    )
    progress_stream = set_progress_tracker_stream(parsed_df, progress)

    display_func = create_display_function(progress, query_processor)

    # Start business processing stream
    results_stream = (
        parsed_df.writeStream.outputMode("append")
        .foreachBatch(display_func)
        .trigger(processingTime="1 seconds")
        .start()
    )

    stratum_stream = None
    if globals().get("sample_config") == "dynamic":
        # Dynamic sampling mode
        def stratum_display_func(batch_df, batch_id):
            global stratum_info
            batch_rows = batch_df.collect()

            # with open(f"logs/dynamic/batch_rows.json", "a") as f:
            #     for row_dict in batch_rows:
            #         f.write(json.dumps(row_dict) + "\n")

            if not batch_rows:
                return

            latest_timestamp = None
            latest_rows = []

            # Find the latest timestamp
            for row in batch_rows:
                row_dict = row.asDict()
                current_timestamp = row_dict.get("timestamp")
                if current_timestamp and (
                    latest_timestamp is None or current_timestamp > latest_timestamp
                ):
                    latest_timestamp = current_timestamp

            if latest_timestamp:
                latest_rows = [
                    row.asDict()
                    for row in batch_rows
                    if row.asDict().get("timestamp") == latest_timestamp
                ]

            # Update global data structure
            stratum_info = {}
            for row_dict in latest_rows:
                stratum_info[row_dict.get("class_label")] = row_dict

            # Write to local file
            # with open(f"logs/base/arxiv_stratum_info.json", "a") as f:
            #     for row_dict in latest_rows:
            #         row_dict["timestamp"] = str(row_dict["timestamp"])
            #         f.write(json.dumps(row_dict) + "\n")

        stratum_df = setup_stratum_stream(
            spark, kafka_stratum_topic, kafka_bootstrap_servers
        )
        stratum_stream = (
            stratum_df.writeStream.outputMode("append")
            .foreachBatch(stratum_display_func)
            .trigger(processingTime="1 seconds")
            .start()
        )

    logger.info(f"Started streaming from topic: {kafka_topic}")
    return results_stream, progress_stream, stratum_stream


def load_query_module(module_name: str):

    try:
        module_path = f"query.{module_name}"
        module = importlib.import_module(module_path)
        logger.info(f"Successfully loaded query module: {module_path}")

        # Load variables from the module into the global namespace
        required_vars = [
            "QUERY",
            "DATA_PATH",
            "PYSPARK_SCHEMA",
            "query_processor",
            "OutputSchema",
        ]

        # Load required variables into the global namespace
        for var_name in required_vars:
            if hasattr(module, var_name):
                globals()[var_name] = getattr(module, var_name)
            else:
                raise AttributeError(
                    f"Module {module_path} is missing required variable: {var_name}"
                )

        # Load optional variables
        if hasattr(module, "vectorization"):
            globals()["vectorization"] = getattr(module, "vectorization")
        if hasattr(module, "SAMPLE_CONFIG"):
            globals()["sample_config"] = getattr(module, "SAMPLE_CONFIG")
        if hasattr(module, "INPUT_OPTIMIZATION"):
            globals()["input_optimizaion"] = getattr(module, "INPUT_OPTIMIZATION")
        if hasattr(module, "OUTPUT_OPTIMIZATION"):
            globals()["output_optimizaion"] = getattr(module, "OUTPUT_OPTIMIZATION")
        if hasattr(module, "FIELDS"):
            globals()["fields"] = getattr(module, "FIELDS")
        if hasattr(module, "DYNAMIC_MODE"):
            globals()["dynamic_mode"] = getattr(module, "DYNAMIC_MODE")

        return module
    except Exception as e:
        logger.error(f"Failed to load query module {module_name}: {e}")
        raise


def main():
    global PROXY_SERVER_URL
    """Run PySpark client"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PySpark Online Aggregation Client")
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help="Name of the query module to use (without path and suffix)",
    )
    parser.add_argument(
        "-p",
        "--proxy-url",
        default=PROXY_SERVER_URL,
        help="URL of the proxy server (e.g., http://localhost:9981/query)",
    )
    args = parser.parse_args()

    # Set global variables from command-line arguments
    PROXY_SERVER_URL = args.proxy_url

    # Load the specified query module
    load_query_module(args.query)

    # Create Spark session
    spark = SparkSession.builder.appName("Online Aggregation Client").getOrCreate()

    # Send query request to proxy server
    response = send_query_to_proxy(QUERY, str(uuid.uuid4()))
    if response.get("status") == "error":
        logger.error(f"Query failed: {response.get('message')}")
        return

    # Get Kafka connection information
    kafka_topic = response.get("kafka_topic")
    kafka_stratum_topic = response.get("kafka_stratum_topic")
    kafka_bootstrap_servers = response.get("kafka_bootstrap_servers")
    total_data_size = response.get("total_data_size", 0)
    logger.info(f"Processing {total_data_size} files")

    # Start Kafka consumer to process data
    results_stream, progress_stream, stratum_stream = start_kafka_consumer(
        spark,
        kafka_topic,
        kafka_stratum_topic,
        kafka_bootstrap_servers,
        total_data_size,
    )

    # Wait for stream processing to complete
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Terminating stream processing...")
        results_stream.stop()
        progress_stream.stop()
        if stratum_stream:
            stratum_stream.stop()
        spark.stop()


if __name__ == "__main__":
    main()
