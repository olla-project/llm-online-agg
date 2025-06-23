from pyspark.sql.functions import col, when, count, lit, sum, expr
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from query.statistical_udf import make_hoeffding_udf
from pydantic import BaseModel
from typing import Literal
import io

# 查询：将电影评论分类为不同的类别
QUERY = (
    "Classify this movie review into one of the following categories: "
    "'positive', 'negative', 'neutral'. "
    "Return only the category name as your answer."
)
DATA_PATH = "../data/final_movies_dataset.csv" # replace by absolute path
# FIELDS = ["reviewcontent"]
SAMPLE_CONFIG = "random"
# SAMPLE_CONFIG = "dynamic"
INPUT_OPTIMIZATION = True
# OUTPUT_OPTIMIZATION = "none"
OUTPUT_OPTIMIZATION = "groupby"
# DYNAMIC_MODE = "random"


class OutputSchema(BaseModel):

    result: Literal["positive", "negative", "neutral"]


PYSPARK_SCHEMA = StructType(
    [
        StructField("result", StringType(), True),
        # StructField("_idx", StringType(), True),
    ]
)


def query_processor(df):
    if df.count() == 0:
        return df

    df_normalized = df.withColumn("category", expr("lower(result)"))

    category_counts = (
        df_normalized.groupBy("category").count().withColumnRenamed("count", "category_count")
    )

    total_count = category_counts.agg({"category_count": "sum"}).collect()[0][0]

    result_df = (
        category_counts.withColumn("total_count", lit(total_count))
        .withColumn("proportion", col("category_count") / col("total_count"))
        .orderBy(col("proportion").desc())
    )

    return result_df
