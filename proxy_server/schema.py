from pydantic import BaseModel
from typing import Literal, Dict, Any, List, Optional
from aiokafka import AIOKafkaProducer


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    request_id: str = None
    data_path: str = None
    query: str
    fields: Optional[list[str]] = None
    output_json_schema: dict = None
    vectorization: Optional[bool] = False
    sample_config: Optional[Literal["none", "random", "filter", "dynamic"]] = None
    input_optimizaion: bool = False
    output_optimization: Literal["select", "where", "groupby", "none"] = "none"
    dynamic_mode: Literal["adjust", "no_adjust", "random"] = "adjust"
    mode: Literal["json", "dspy"] = "json"  # deprecation: dspy related
    signature_def: Optional[Dict[str, Any]] = None  # deprecation: dspy related


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    status: str
    message: str
    request_id: str
    total_data_size: int
    kafka_topic: str
    kafka_stratum_topic: str = None
    kafka_bootstrap_servers: str


class ProcessingContext(BaseModel):
    request: QueryRequest
    producer: Any  # AIOKafkaProducer
    topic: str
    stratum_topic: str = None
