import pandas as pd
import numpy as np
from text_index import FilterTextIndex, DynamicTextIndex
from schema import ProcessingContext
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SamplingStrategy:
    def sample(self, data: pd.DataFrame, context: ProcessingContext):
        raise NotImplementedError


# Concrete strategy implementations
class NoSampling(SamplingStrategy):
    def sample(self, data: pd.DataFrame, context: ProcessingContext):
        return data


class RandomSampling(SamplingStrategy):
    def sample(self, data: pd.DataFrame, context: ProcessingContext):
        return data.sample(frac=1.0)


class FilterSampling(SamplingStrategy):

    def __init__(self):
        self.cluster_distribution = None
        self.sample_size = 512  # Default sample size for optimization

    def build_inedx(self, data: pd.DataFrame, context: ProcessingContext):
        # Combine fields into a single text representation
        field_texts = []
        for _, row in data.iterrows():
            row_text = " ".join(
                [
                    str(row.get(field, ""))
                    for field in context.request.fields
                    if field in row
                ]
            )
            field_texts.append(row_text)
        text_index = FilterTextIndex(field_texts, n_clusters=2)
        cluster_ids = [text_index.get_cluster(text) for text in field_texts]
        # data = data.copy()  # Create a copy to avoid modifying original data
        data["cluster_id"] = cluster_ids

        cluster_distribution = text_index.get_cluster_distribution()
        logger.info(f"Cluster distribution: {cluster_distribution}")
        self.cluster_distribution = cluster_distribution
        return data

    def sample(self, data: pd.DataFrame):
        sampled_data = pd.DataFrame()

        total_samples = sum(self.cluster_distribution.values())
        for cluster_id, count in self.cluster_distribution.items():
            # Calculate sample size for this cluster
            cluster_sample_size = max(
                64, round(self.sample_size * count / total_samples)
            )
            cluster_sample_size = max(cluster_sample_size, round(count * 0.1))

            # Sample data from this cluster
            cluster_data = data[data["cluster_id"] == cluster_id]
            if len(cluster_data) > cluster_sample_size:
                sampled_cluster_data = cluster_data.sample(n=cluster_sample_size)
            else:
                sampled_cluster_data = cluster_data
            logger.info(
                f"Sampled {len(sampled_cluster_data)} data points for cluster {cluster_id}"
            )
            sampled_data = pd.concat([sampled_data, sampled_cluster_data])

        # Mark sampled data to avoid re-processing later
        sampled_indices = sampled_data.index
        data["is_sampled"] = False
        data.loc[sampled_indices, "is_sampled"] = True

        return sampled_data, data

    def adjust(
        self, data: pd.DataFrame, sampled_data: pd.DataFrame, sample_results: list
    ):
        # Count the processing results for each cluster_id
        true_counts = {cluster_id: 0 for cluster_id in self.cluster_distribution.keys()}
        total_counts = {
            cluster_id: 0 for cluster_id in self.cluster_distribution.keys()
        }

        # Based on the implementation of the processing functions, input and output order is consistent
        # The output results also contain the original data, so index relationships can be used directly
        for i, result in enumerate(sample_results):
            cluster_id = result["cluster_id"]
            if result["result"]:
                true_counts[cluster_id] += 1
            total_counts[cluster_id] += 1

        # Calculate the true proportion for each cluster_id
        true_ratios = {}
        for cluster_id in self.cluster_distribution.keys():
            if total_counts[cluster_id] > 0:
                true_ratios[cluster_id] = (
                    true_counts[cluster_id] / total_counts[cluster_id]
                )
            else:
                true_ratios[cluster_id] = 0.0

        sorted_clusters = sorted(true_ratios.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Filter strategy sorted_clusters: {sorted_clusters}")

        remaining_data = data[~data["is_sampled"]].copy()

        if remaining_data.empty:
            return []

        # Filter the remaining data for each cluster_id in sorted_clusters order to form list[df]
        df_list = []
        for cluster_id, _ in sorted_clusters:
            cluster_df = remaining_data[
                remaining_data["cluster_id"] == cluster_id
            ].copy()
            # Shuffle the data
            cluster_df = cluster_df.sample(frac=1.0)
            if not cluster_df.empty:
                df_list.append(cluster_df)
            logger.info(f"Cluster_id: {cluster_id}, remaining_data: {len(cluster_df)}")
        # Only return clusters with data, ignoring any unassigned cluster_ids
        return df_list


class DynamicSampling(SamplingStrategy):

    def __init__(self):
        self.text_index: DynamicTextIndex = None

    def build_inedx(self, data: pd.DataFrame, context: ProcessingContext):
        # Combine fields into a single text representation
        field_texts = []
        for _, row in data.iterrows():
            row_text = " ".join(
                [
                    str(row.get(field, ""))
                    for field in context.request.fields
                    if field in row
                ]
            )
            field_texts.append(row_text)
        self.text_index = DynamicTextIndex(
            field_texts,
            n_classes=5,
            n_strata=(1 if context.request.dynamic_mode == "random" else None),
        )
        return data

    def sample(
        self, data: pd.DataFrame, sample_size: int
    ) -> tuple[list[pd.DataFrame], list[np.array]]:
        sampled_indices: list[np.array] = self.text_index.sample(sample_size)
        sampled_data: list[pd.DataFrame] = []
        for indices in sampled_indices:
            if len(indices) == 0:
                sampled_data.append(pd.DataFrame())
            else:
                sub_data = data.iloc[indices]
                sampled_data.append(sub_data)
        return sampled_data, sampled_indices

    def adjust(
        self,
        sample_results: list[list[dict]],
        sampled_indices: list[np.array],
        data: pd.DataFrame,
    ) -> tuple[dict, pd.DataFrame, Optional[list[int]]]:
        category_results = []
        for strata_results in sample_results:
            category_result = []
            for result in strata_results:
                category_result.append(result["result"])
            category_results.append(np.array(category_result))

        all_updated_data_indices = self.text_index.adjust()

        return {}, data, None
