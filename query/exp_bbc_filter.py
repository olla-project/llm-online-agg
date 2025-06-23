from pyspark.sql.functions import col, when, avg, count, lit, split, var_samp
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType,
    BooleanType,
)
from typing import Literal
from pydantic import BaseModel
from query.statistical_udf import make_hoeffding_udf, make_large_sample_udf

QUERY = "Determine whether the news content is related to sports"
DATA_PATH = "../data/BBC_News_Train_with_view_like_x10.csv" # replace by absolute path
INPUT_OPTIMIZATION = True
OUTPUT_OPTIMIZATION = "none"
SAMPLE_CONFIG = "filter"


class OutputSchema(BaseModel):
    result: bool


PYSPARK_SCHEMA = StructType(
    [
        StructField("result", BooleanType(), True),
        StructField("timestamp", StringType(), True),
        StructField("LikeCount", IntegerType(), True)

    ]
)


def query_processor(df):
    filtered_df = df.filter(col("result") == True)

    avg_like = avg("LikeCount")
    var_like = var_samp("LikeCount")
    count_like = count("LikeCount")

    stats_df = filtered_df.agg(
        avg_like.alias("avg_like"),
        var_like.alias("like_variance"),
        count_like.alias("like_count")
    )

    large_sample_udf = make_large_sample_udf(p=0.95)
    stats_df = stats_df.withColumn(
        "confidence_interval",
        large_sample_udf(col("like_count"), col("like_variance"))
    )

    result = stats_df.select("avg_like", "confidence_interval")
    return result
