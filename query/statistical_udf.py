from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import math
from typing import List
from scipy.stats import norm


def hoeffding_interval(n: int, delta=0.05, a=0.0, b=1.0) -> float:
    """计算Hoeffding置信区间

    Args:
        n: 样本数量
        delta: 置信度参数 (1-delta是置信度)
        a, b: 数值范围

    Returns:
        epsilon: 置信区间半径
    """
    if n == 0:
        return 0.0

    # 计算置信区间半径
    epsilon = math.sqrt(((b - a) ** 2) * math.log(2 / delta) / (2 * n))
    return epsilon


def make_hoeffding_udf(delta=0.05, a=0.0, b=1.0):
    return udf(lambda n: hoeffding_interval(n, delta, a, b), DoubleType())


def get_z(p: float) -> float:
    # 对称双尾 (100p)% CI，对应的单侧累积概率 (1+p)/2
    # Hardcoded common values to avoid using scipy.stats.norm in UDF
    if p == 0.95:
        return 1.96  # Z-score for 95% confidence
    elif p == 0.99:
        return 2.576  # Z-score for 99% confidence
    elif p == 0.90:
        return 1.645  # Z-score for 90% confidence
    else:
        # Only use norm.ppf at definition time, not execution time
        return norm.ppf((1 + p) / 2)


def make_large_sample_udf(p: float = 0.95):
    """
    创建一个用于计算大样本置信区间的UDF

    Args:
        p: 置信度, 默认为0.95对应95%置信区间

    Returns:
        用于计算置信区间的UDF
    """
    # Calculate the z-score at definition time, not execution time
    z_value = get_z(p)

    # Define a simple function that doesn't use any external libraries
    def calc_interval(n, s2):
        if n < 2:
            return 0.0
        return z_value * math.sqrt(s2 / n)

    # Use the simple Python function in the UDF
    return udf(calc_interval, DoubleType())
