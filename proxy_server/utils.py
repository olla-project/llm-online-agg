from typing import Dict, Any, List
import json
from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import logging
from schema import QueryRequest
import pandas as pd
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI
import os
from pydantic import BaseModel
import re

logger = logging.getLogger(__name__)

LOG_DIR = Path(os.getenv("LOG_DIR", ""))


async def publish_to_kafka(
    producer: AIOKafkaProducer,
    result: Dict[str, Any],
    topic: str,
    add_timestamp: bool = True,
) -> bool:
    try:
        if add_timestamp:
            result["timestamp"] = datetime.now().isoformat()
        data = json.dumps(result).encode("utf-8")
        # print(f"Publishing to Kafka topic: {topic}, data: {data}")
        await producer.send_and_wait(topic, data)
        return True
    except Exception as e:
        logger.error(f"Error publishing to Kafka: {e}")
        return False


async def ensure_topic_exists(admin_client: AIOKafkaAdminClient, topic: str) -> bool:
    """Ensure a Kafka topic exists by creating it with the Admin API.

    Args:
        admin_client: The Kafka admin client instance
        topic: The Kafka topic name to create

    Returns:
        bool: True if successful, False otherwise
    """
    assert admin_client is not None, "Kafka admin client not initialized"

    try:
        # Create topic configuration
        topic_config = NewTopic(
            name=topic,
            num_partitions=1,  # Set to 1 partition for simplicity
            replication_factor=1,  # Set to 1 for local development
        )

        # Create the topic
        await admin_client.create_topics([topic_config])
        logger.info(f"Created Kafka topic: {topic}")
        return True
    except TopicAlreadyExistsError:
        # This is not an error, the topic already exists
        logger.info(f"Kafka topic already exists: {topic}")
        return True
    except Exception as e:
        logger.error(f"Error creating Kafka topic {topic}: {e}")
        return False


def get_data(request: QueryRequest) -> pd.DataFrame:
    data_path = request.data_path
    assert data_path is not None, "data_path must be specified"

    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    elif data_path.endswith(".xlsx"):
        data = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    if "reviewtype" in data.columns:
        data = data.drop(columns=["reviewtype"])
    if "Category" in data.columns:
        data = data.drop(columns=["Category"])
    if "categories" in data.columns:
        data = data.drop(columns=["categories"])
    if "extracted_data" in data.columns:
        data = data.drop(columns=["extracted_data"])
    return data


async def log_llm_result_to_jsonl(idx: int, result: Dict[str, Any], request_id: str):
    timestamp = datetime.now().isoformat()
    log_entry = {"timestamp": timestamp, "idx": idx, "result": result}

    # Create log file name using request ID
    log_file = LOG_DIR / f"llm_results_{request_id}.jsonl"

    try:
        # Ensure log directory exists
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # Write to JSONL in append mode
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        logger.debug(f"Logged result for index {idx} to {log_file}")
    except Exception as e:
        logger.error(f"Error logging to JSONL: {e}")


async def input_fields_optimization(query: str, data: pd.DataFrame) -> list[str]:
    """
    Call LLM to return a list of field names related to the query based on the query and data sample.
    """

    # 取前 3 行作为 sample，防止 prompt 太长
    data_sample = data.head(1).to_dict(orient="records")
    columns = list(data.columns)

    # Compose system prompt in English
    system_content = (
        "You are a data analysis assistant. The user will give you a natural language question (query) and a data sample (each row is a record, columns are field names). "
        "Based on the query, decide which fields (columns) in the data sample are necessary to answer the question. "
        "Only return the list of required field names in the data sample, and nothing else. "
        "If some fields are just IDs, indices, or irrelevant to the question, do not include them. "
        "Your output must strictly follow the provided JSON schema."
    )
    user_content = (
        f"User query: {query}\n"
        f"Data sample: {json.dumps(data_sample, ensure_ascii=False)}\n"
        f"All fields: {columns}\n"
        "Please only return the required field names in the data sample as a list."
    )

    # 构造 pydantic schema
    class InputFieldsOptimizationOutput(BaseModel):
        fields: list[str]

    json_schema = InputFieldsOptimizationOutput.model_json_schema()

    client = AsyncOpenAI(
        base_url=os.getenv("LLM_BASE_URL"), api_key="dummy-key", timeout=30
    )
    response = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=256,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": json_schema.get("title", "InputFieldsOptimizationOutput"),
                "schema": json_schema,
            },
        },
    )
    # 解析 LLM 返回
    content = response.choices[0].message.content
    try:
        result = InputFieldsOptimizationOutput.model_validate_json(content)
        logger.info(f"LLM optimization fields: {result.fields}")
        return result.fields
    except Exception as e:
        # fallback: try to parse directly as list[str]
        try:
            arr = json.loads(content)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
        raise RuntimeError(f"LLM output parsing failed: {content}, error: {e}")


def generate_regex_from_schema(json_schema: Dict[str, Any]) -> str:
    """
    Convert Pydantic-generated JSON Schema to a simple key=value regular expression format.
    Assumes schema defines properties and supports enum types.

    Returns a complete regex pattern for validating the entire LLM output.
    """
    props = json_schema.get("properties", {})
    patterns = []
    for name, spec in props.items():
        if "enum" in spec:
            # Enumeration type
            options = spec["enum"]
            # Escape each option
            opts = [re.escape(str(opt)) for opt in options]
            pat = f"{name}=(?:{'|'.join(opts)})"
        else:
            # Simplified handling, support basic types: string and number
            type_ = spec.get("type", "string")
            if type_ == "string":
                pat = f"{name}=([^;]+)"
            elif type_ in ("integer", "number"):
                pat = f"{name}=([-+]?[0-9]*\.?[0-9]+)"
            else:
                # fallback for other types
                pat = f"{name}=([^;]+)"
        patterns.append(pat)
    # Multiple fields separated by semicolons
    full = ";".join(patterns)
    return full


def parse_structured_response(response: str) -> Dict[str, Any]:
    """
    Parse a string in the format key=value;key2=value2 into a dictionary.
    Automatically convert numeric strings to int or float.
    """
    result: Dict[str, Any] = {}
    # Split by semicolons
    parts = [seg.strip() for seg in response.split(";") if seg.strip()]
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        # Try to convert to numbers
        if re.fullmatch(r"[-+]?[0-9]+", val):
            result[key] = int(val)
        elif re.fullmatch(r"[-+]?[0-9]*\.?[0-9]+", val):
            result[key] = float(val)
        else:
            result[key] = val
    return result
