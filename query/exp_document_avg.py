from pyspark.sql.functions import col, when, avg, count, lit, split, var_samp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from typing import Literal
from pydantic import BaseModel
from query.statistical_udf import make_large_sample_udf

QUERY = "What's the total price and order_id of this document?"
DATA_PATH = "../data/company_document_x4.csv" # replace by absolute path
# FIELDS = ["file_content"]
INPUT_OPTIMIZATION = False
OUTPUT_OPTIMIZATION = "none"
# OUTPUT_OPTIMIZATION = "select"
SAMPLE_CONFIG = "random"

class OutputSchema(BaseModel):
    total_price: float
    # order_id: str


PYSPARK_SCHEMA = StructType(
    [
        StructField("total_price", FloatType(), True),
        # StructField("order_id", StringType(), True),
        StructField("document_type", StringType(), True),
        StructField("order_date", StringType(), True),

    ]
)


def query_processor(df):
    if "order_date" in df.columns:
        df = df.withColumn("year", split(col("order_date"), "-").getItem(0).cast("int"))
    
    large_sample_udf = make_large_sample_udf(p=0.95)
    
    grouped_stats = df.groupBy("document_type").agg(
        count("total_price").alias("sample_count"),
        avg(col("total_price").cast("double")).alias("avg_price"),
        var_samp(col("total_price").cast("double")).alias("price_variance")
    )
    
    result = grouped_stats.withColumn(
        "confidence_interval", 
        large_sample_udf(col("sample_count"), col("price_variance"))
    )
    
    result = result.orderBy(col("avg_price").desc())

    result = result.select("document_type", "sample_count", "avg_price", "confidence_interval")
    
    return result
