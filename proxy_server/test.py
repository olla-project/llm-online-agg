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
 
openai_client = AsyncOpenAI(
    base_url=LLM_BASE_URL,  # Extract base URL from the API URL
    api_key="dummy-key",  # Placeholder for local setup
)

# test chat 
async def test_chat():
    user_prompt = "hello"
    response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(test_chat())
    