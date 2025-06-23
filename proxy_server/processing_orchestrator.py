from sampling_strategy import (
    NoSampling,
    RandomSampling,
    FilterSampling,
    DynamicSampling,
)
from data_processor import DataProcessor
import logging
import pandas as pd
import numpy as np
import asyncio
from schema import ProcessingContext
import time

logger = logging.getLogger(__name__)


class ProcessingOrchestrator:

    @classmethod
    async def process_with_strategy(cls, strategy, data, context):
        logger.info(f"typeof strategy: {type(strategy)}")
        if isinstance(strategy, (NoSampling, RandomSampling)):
            sampled_data = strategy.sample(data, context)
            logger.info(f"length of sampled_data: {len(sampled_data)}")
            return await DataProcessor.process(sampled_data, strategy, context)
        elif isinstance(strategy, FilterSampling):
            return await cls._process_with_filter_sampling(strategy, data, context)
        elif isinstance(strategy, DynamicSampling):
            return await cls._process_with_dynamic_sampling(strategy, data, context)

    @classmethod
    async def _process_with_filter_sampling(cls, strategy, data, context):
        data = strategy.build_inedx(data, context)
        sampled_data, data = strategy.sample(data)
        # logger.info(f"type of sampled_data: {type(sampled_data)}")
        sampl_results = await DataProcessor.process(sampled_data, strategy, context)
        adjuseted_remaining_data_list = strategy.adjust(
            data, sampled_data, sampl_results
        )
        # logger.info(f"type of adjuseted_remaining_data: {type(adjuseted_remaining_data)}")
        remaining_results = []
        for adjuseted_remaining_data in adjuseted_remaining_data_list:
            remaining_results.extend(
                await DataProcessor.process(adjuseted_remaining_data, strategy, context)
            )

        return sampl_results + remaining_results

    @classmethod
    async def _process_with_dynamic_sampling(
        cls, strategy: DynamicSampling, data: pd.DataFrame, context: ProcessingContext
    ):
        total_size = remaining_size = len(data)
        data = strategy.build_inedx(data, context)

        iteration = 0
        while remaining_size > 0:
            iteration += 1
            current_sample_size = min(int(total_size * 0.1), remaining_size)
            if current_sample_size == 0:
                break
            sampled_data, sampled_indices = strategy.sample(
                data, current_sample_size
            )  # list[pd.DataFrame], list[np.array]
            # logger.info(f"iteration: {iteration}, sampled_indices: {sampled_indices}")
            tasks = []
            for stratum_idx, strata_sampled_data in enumerate(sampled_data):
                tasks.append(
                    DataProcessor.process(
                        strata_sampled_data, strategy, context, stratum_idx
                    )
                )
            sample_results = await asyncio.gather(*tasks)

            if context.request.dynamic_mode == "adjust":
                adjust_start_time = time.time()
                intervals, data, republish_indices = strategy.adjust(
                    sample_results, sampled_indices, data
                )
                adjust_end_time = time.time()
                logger.info(f"Adjust time: {adjust_end_time - adjust_start_time}")
            remaining_size -= current_sample_size
