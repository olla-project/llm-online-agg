import logging
import json
from typing import Dict, Any, List, Optional, Literal
import asyncio
from pydantic import ValidationError
import pandas as pd
from schema import QueryRequest, ProcessingContext
from aiokafka import AIOKafkaProducer
import os
from utils import (
    publish_to_kafka,
    log_llm_result_to_jsonl,
    generate_regex_from_schema,
    parse_structured_response,
)
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from sampling_strategy import (
    SamplingStrategy,
    NoSampling,
    RandomSampling,
    FilterSampling,
    DynamicSampling,
)
import dotenv
from datetime import datetime
import time
import threading

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def watchdog(timeout, context=None):
    """Prints a warning message if not cancelled within the specified time period."""

    def warning():
        print(f"[WARNING] Process stuck: {context}")

    timer = threading.Timer(timeout, warning)
    timer.start()
    return timer


MODEL_NAME = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
data_processing_semaphore = asyncio.Semaphore(int(os.getenv("CONCURRENCY_LIMIT", "10")))


class DataProcessor:

    @classmethod
    async def process(
        cls,
        data: pd.DataFrame,
        strategy: SamplingStrategy,
        context: ProcessingContext,
        stratum_idx: Optional[int] = None,  # Only used in dynamic sampling
    ) -> List[Dict[str, Any]]:
        if data.empty:
            return []

        if isinstance(strategy, DynamicSampling):
            assert (
                stratum_idx is not None
            ), "stratum_idx must be provided for dynamic sampling"

        logger.info(f"Semaphore remaining value: {data_processing_semaphore._value}")

        request, kafka_producer, kafka_topic, stratum_topic = (
            context.request,
            context.producer,
            context.topic,
            context.stratum_topic,
        )

        # logger.info(f"request: {request}")

        # Check output JSON schema
        json_schema = request.output_json_schema
        if not json_schema:
            raise ValueError("output_json_schema is required for json mode")

        regex_schema = None
        if request.output_optimization != "none":
            if request.output_optimization == "where":
                regex_schema = "(true|false)"
            elif request.output_optimization == "select":
                regex_schema = generate_regex_from_schema(json_schema)
            elif request.output_optimization == "groupby":
                regex_schema = generate_regex_from_schema(json_schema).split("=")[1]

        logger.info(f"regex_schema: {regex_schema}")

        all_results = []
        task_counter = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        client = AsyncOpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key="dummy-key")

        async def process_atomic_item(item_data: Dict[str, Any]):
            if not item_data:
                return None

            async with data_processing_semaphore:
                nonlocal task_counter
                task_counter += 1
                # logger.info(f"[Task {task_counter}] Acquired semaphore and started processing, semaphore remaining value: {data_processing_semaphore._value}")

                timer = watchdog(10, "_process_item")
                result = await cls._process_item(
                    client,
                    request.query,
                    item_data,
                    request.fields,
                    json_schema,
                    request.output_optimization,
                    regex_schema,
                )
                timer.cancel()

            result_data, result_usage = result["output"], result["usage"]
            total_usage["prompt_tokens"] += result_usage.prompt_tokens
            total_usage["completion_tokens"] += result_usage.completion_tokens
            total_usage["total_tokens"] += result_usage.total_tokens

            post_process_task = []
            if kafka_producer and result_data:
                post_process_task.append(
                    publish_to_kafka(kafka_producer, result_data, kafka_topic)
                )
            if result_data:
                post_process_task.append(
                    log_llm_result_to_jsonl(
                        task_counter, result_data, request.request_id
                    )
                )
            if isinstance(strategy, DynamicSampling):
                # TODO: Assuming the output schema of the classification LLM task contains a 'result' field
                strategy.text_index.record(
                    stratum_id=stratum_idx,
                    idx=item_data["_idx"],
                    result=result_data["result"],
                )
                interval_res = strategy.text_index.confidence_intervals
                if interval_res:
                    for res in interval_res:
                        post_process_task.append(
                            publish_to_kafka(
                                kafka_producer, res, stratum_topic, add_timestamp=False
                            )
                        )

            timer = watchdog(10, "post_process_task")
            await asyncio.gather(*post_process_task)
            timer.cancel()
            return result_data

        start_time = time.time()
        all_tasks = [
            process_atomic_item(item_data)
            for item_data in data.to_dict(orient="records")
        ]
        results = []
        # WARNING as_completed can't ensure the order of results, but it's ok for us, because we will not use the results later
        for coro in tqdm.as_completed(
            all_tasks, desc="Processing data", total=len(all_tasks)
        ):
            try:
                result = await coro
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue
        # results = await asyncio.gather(*all_tasks)
        all_results = [r for r in results if r is not None]

        logger.info(
            f"DataProcessor complete, data size: {len(all_results)}, usage: {total_usage}, time: {time.time() - start_time}"
        )

        if isinstance(strategy, FilterSampling):
            # 统计 result 字段为 True 的比例
            true_count = sum(1 for result in all_results if result.get("result", False))
            logger.info(
                f"Filter Sampling Cluster_id: {all_results[0].get('cluster_id', None)}, True ratio: {true_count / len(all_results)}"
            )
        return all_results

    @classmethod
    async def _process_item(
        cls,
        client: AsyncOpenAI,
        query: str,
        data_item: Dict[str, Any],
        fields: List[str],
        json_schema: Dict[str, Any],
        output_optimization: Literal["select", "where", "groupby", "none"] = "none",
        regex_schema: str = None,
    ) -> Dict[str, Any]:
        # Select only the required fields
        if fields:
            item_data = {
                field: data_item[field] for field in fields if field in data_item
            }
        else:
            item_data = data_item

        def json_schema_compressor(json_schema: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "properties": {
                    k: v["type"] for k, v in json_schema["properties"].items()
                }
            }

        try:
            # Construct the prompt
            system_content, user_content = None, None
            response = None
            if output_optimization == "none":
                system_content = "You are a helpful AI assistant that processes data according to user queries. Your answers should be precise, relevant and follow the JSON format provided by user without any additional information."
                user_content = f"Query: {query}\nData: {json.dumps(item_data, ensure_ascii=False)}\nOutput json schema: {json.dumps(json_schema, ensure_ascii=False)}"
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    max_tokens=256,
                    # sglang structured output
                    # response_format={
                    #     "type": "json_schema",
                    #     "json_schema": {
                    #         "name": json_schema.get("title", "OutputSchema"),
                    #         "schema": json_schema,
                    #     },
                    # },
                    # vllm structured output
                    extra_body={"guided_json": json_schema},
                )
                response_content = response.choices[0].message.content
                result = json.loads(response_content)
            else:
                system_content = "You are a helpful AI assistant that processes data according to user queries. Your answers should be precise, relevant and follow the Regular expression provided by user without any additional information."
                user_content = f"Query: {query}\nData: {json.dumps(item_data, ensure_ascii=False)}\nOutput Regular expression: {regex_schema}"
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    max_tokens=256,
                    # sglang structured output
                    # extra_body={"regex": regex_schema},
                    # vllm structured output
                    extra_body={"guided_regex": regex_schema},
                    # extra_body={"guided_json": json_schema},
                )
                response_content = response.choices[0].message.content
                if output_optimization != "none":
                    if output_optimization == "where":
                        result = {"result": response_content == "true"}
                    elif output_optimization == "select":
                        result = parse_structured_response(response_content)
                    elif output_optimization == "groupby":
                        result = {"result": response_content}

            # logger.info(f"LLM response: {response_content}")

            # Merge LLM results with original data
            output = {}
            output.update(result)
            output.update(data_item)  # Preserve original data
            # logger.info(f"usage: {response.usage}")

            # If usage tracking is not needed:
            # return {"output": output, "usage": None}
            return {"output": output, "usage": response.usage}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}
