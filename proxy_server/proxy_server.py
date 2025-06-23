import json
import logging
import os
import random
import asyncio
import time
import uuid
import importlib
import inspect
import warnings
import json
from datetime import datetime
from typing import (
    Dict,
    Any,
    List,
    Coroutine,
    Literal,
    get_type_hints,
    get_origin,
    get_args,
    Tuple,
)
from pathlib import Path
import aiohttp
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient
from openai import AsyncOpenAI

from schema import QueryRequest, QueryResponse, ProcessingContext
from utils import ensure_topic_exists, get_data, input_fields_optimization
import dotenv

from processing_orchestrator import ProcessingOrchestrator
from sampling_strategy import (
    NoSampling,
    RandomSampling,
    FilterSampling,
    DynamicSampling,
)

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration constants
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_TOPIC_PREFIX = os.getenv("KAFKA_TOPIC_PREFIX")
MAX_DATA_SIZE = int(os.getenv("MAX_DATA_SIZE"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")


# Global resources (initialized during startup)
producer = None
admin_client = None
data_semaphore = None
openai_client = None


@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for startup and shutdown operations.

    Args:
        app: FastAPI application instance
    """
    # Startup logic
    global producer, admin_client, openai_client

    # Initialize OpenAI client
    openai_client = AsyncOpenAI(
        base_url=LLM_BASE_URL,  # Extract base URL from the API URL
        api_key="dummy-key",  # Placeholder for local setup
    )
    logger.info("OpenAI client initialized")

    try:
        # Initialize Kafka producer
        producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        await producer.start()
        logger.info(
            f"Kafka producer initialized, connected to {KAFKA_BOOTSTRAP_SERVERS}"
        )

        # Initialize Kafka admin client
        admin_client = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        await admin_client.start()
        logger.info(
            f"Kafka admin client initialized, connected to {KAFKA_BOOTSTRAP_SERVERS}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Kafka clients: {e}")
        producer = None
        admin_client = None

    yield

    # Shutdown logic
    if producer:
        await producer.stop()
        logger.info("Kafka producer stopped")

    if admin_client:
        await admin_client.close()
        logger.info("Kafka admin client stopped")


# Initialize FastAPI application
app = FastAPI(title="LLM Proxy Server", lifespan=lifespan)


async def process_data_and_publish(
    request: QueryRequest, topic_name: str, stratum_topic_name: str, data: pd.DataFrame
):
    if request.input_optimizaion:
        request.fields = await input_fields_optimization(request.query, data)

    context = ProcessingContext(
        producer=producer,
        topic=topic_name,
        stratum_topic=stratum_topic_name,
        request=request,
        max_data_size=MAX_DATA_SIZE,
    )

    sample_config = request.sample_config
    if sample_config == "dynamic":
        data["_idx"] = data.index
        strategy = DynamicSampling()
    elif sample_config == "none":
        strategy = NoSampling()
    elif sample_config == "random":
        strategy = RandomSampling()
    elif sample_config == "filter":
        strategy = FilterSampling()

    return await ProcessingOrchestrator.process_with_strategy(strategy, data, context)


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Handle a query request from a PySpark client.

    Immediately returns a response with Kafka topic info, then processes files in background.

    Args:
        request: Query request containing query text and optional request ID
        background_tasks: FastAPI background tasks object

    Returns:
        Response with request status and Kafka connection details
    """
    # Generate a request ID if not provided
    logger.info(f"Received query request: {request}")
    if not request.request_id:
        request.request_id = str(uuid.uuid4())

    topic_name = f"{KAFKA_TOPIC_PREFIX}{request.request_id}"
    stratum_topic_name = f"{KAFKA_TOPIC_PREFIX}{request.request_id}_stratum"
    logger.info(
        f"Received query request with ID: {request.request_id}, topic: {topic_name}"
    )

    try:
        data = get_data(request)
        data = data.iloc[:MAX_DATA_SIZE] if len(data) > MAX_DATA_SIZE else data

        # Create the Kafka topic before returning response
        ensure_tasks = []
        # topic_created = await ensure_topic_exists(admin_client, topic_name)
        ensure_tasks.append(ensure_topic_exists(admin_client, topic_name))
        ensure_tasks.append(ensure_topic_exists(admin_client, stratum_topic_name))
        topic_created = await asyncio.gather(*ensure_tasks)
        topic_created = all(
            topic_created
        )  # Check if all topics were created successfully
        if not topic_created:
            logger.error(f"Failed to ensure kafka topic, continuing anyway")

        # Schedule background processing
        background_tasks.add_task(
            process_data_and_publish, request, topic_name, stratum_topic_name, data
        )

        # Return immediate response with connection details
        return QueryResponse(
            status="processing",
            message="Query received, processing files in background",
            request_id=request.request_id,
            total_data_size=len(data),
            kafka_topic=topic_name,
            kafka_stratum_topic=stratum_topic_name,
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        )
    except Exception as e:
        logger.error(f"Error setting up file processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=9981)
