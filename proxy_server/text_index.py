"""
TextIndex class for creating category indexes for a collection of texts.
Uses sentence-transformers embeddings and faiss clustering.
"""

import numpy as np
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Any,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
)
import faiss
from sentence_transformers import SentenceTransformer
import torch
from functools import wraps
import logging
from scipy.stats import norm
from datetime import datetime

logger = logging.getLogger(__name__)


class FilterTextIndex:
    """
    Class to create category indexes for a collection of texts using
    transformer embeddings and unsupervised clustering.
    """

    def __init__(
        self,
        texts: List[str],
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        n_clusters: int = 10,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # self.device = "cpu"

        # Load the sentence transformer model
        self.model = SentenceTransformer(
            model_name,
            cache_folder="/fs/fast/share/pingtai_cc/models/huggingface/",
            local_files_only=True,
            device=self.device,
        )

        # Store text to cluster mapping
        self.text_to_cluster = {}
        self.texts = []
        self.embeddings = None

        self._fit(texts)

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        # Process in batches using sentence-transformers' built-in batching
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device,
        )

        return embeddings

    def _fit(self, texts: List[str]) -> "FilterTextIndex":
        self.texts = texts

        embeddings = self._get_embeddings(texts)
        self.embeddings = embeddings

        dimension = embeddings.shape[1]

        kmeans = faiss.Kmeans(
            dimension, self.n_clusters, niter=20, verbose=True, seed=self.random_state
        )

        embeddings = np.ascontiguousarray(embeddings.astype("float32"))

        kmeans.train(embeddings)

        _, cluster_indices = kmeans.index.search(embeddings, 1)
        cluster_indices = cluster_indices.flatten()

        self.text_to_cluster = {
            text: int(cluster) for text, cluster in zip(texts, cluster_indices)
        }

        return self

    def get_cluster(self, text: str) -> Optional[int]:
        """Get the cluster index for a given text."""
        if text not in self.text_to_cluster:
            logger.warning(f"Text '{text}' not found in the index.")
            return None
        return self.text_to_cluster[text]

    def get_cluster_distribution(self) -> Dict[int, int]:
        distribution = {}
        for cluster in self.text_to_cluster.values():
            distribution[cluster] = distribution.get(cluster, 0) + 1
        return distribution


T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


class CacheProperty(Generic[T]):
    """Property decorator for caching attribute values"""

    def __init__(self, func: Callable[..., T]):
        self.func = func
        self.name = func.__name__
        self._cache_name = f"_{func.__name__}_cache"

    def __get__(self, instance, owner) -> T:
        if not hasattr(instance, self._cache_name):
            cache = self.func(instance)
            setattr(instance, self._cache_name, cache)
        return getattr(instance, self._cache_name)

    def __delete__(self, instance):
        """Delete the cache"""
        if hasattr(instance, self._cache_name):
            delattr(instance, self._cache_name)


def invalidate_cache(func: Callable[P, R]) -> Callable[P, R]:
    """Function decorator that invalidates all CacheProperty caches when the decorated method is called"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for attr in self.__class__.__dict__.values():
            if isinstance(attr, CacheProperty):
                attr.__delete__(self)
        return func(self, *args, **kwargs)

    return wrapper


class Stratum:
    """Represents a data stratum that contains sample indices, embedding vectors, and label information"""

    def __init__(self, indices: np.ndarray, embedding: np.ndarray, alpha: float):
        self.indices = indices  # shape: (n_samples,)
        self.embedding = embedding  # shape: (dimension,)
        self.known_dict: dict[int, Any] = (
            {}
        )  # Record known sample results {index: label}
        # Record exponentially smoothed variance of classes {label: smoothed_variance}
        self.smoothed_var_dict: dict[Any, float] = {}

        self.alpha = alpha  # Exponential smoothing
        self.n = 0  # Iteration count

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample a specified number of samples from this stratum, ensuring not to re-sample already sampled samples"""
        available_indices = [idx for idx in self.indices if idx not in self.known_dict]

        if self.available_size <= n_samples:
            return np.array(available_indices)

        sampled = np.random.choice(available_indices, size=n_samples, replace=False)
        return sampled

    @invalidate_cache  # Update known sample results
    def record(self, idx: int, result: Any) -> None:
        """Update known sample results"""
        self.known_dict[idx] = result

    @invalidate_cache  # Delete known sample results
    def delete(self, idx: int) -> None:
        """Delete known sample results"""
        if idx in self.known_dict:
            del self.known_dict[idx]

    def get_cls_var(self, cls: Any) -> float:
        """Calculate the variance of a specified class in this stratum"""
        var = self.var_dict.get(cls, 0)
        old_var = self.smoothed_var_dict.get(cls, 0)

        self.smoothed_var_dict[cls] = (1 - self.alpha) * var + self.alpha * old_var
        self.n += 1

        return self.smoothed_var_dict[cls] / (1 - self.alpha**self.n)

    # Get the variance contribution dictionary for each class
    @CacheProperty
    def var_dict(self) -> Dict[Any, float]:
        """Get the variance contribution dictionary for each class"""
        if self.known_size == 0:
            return {}

        return {
            cls: self._cal_var(proportion)
            for cls, proportion in self.proportions.items()
        }

    @CacheProperty
    def tag(self) -> Any:
        if self.known_size == 0:
            return None
        return max(self.proportions, key=self.proportions.get)

    @CacheProperty
    def proportions(self) -> dict[Any, float]:
        if self.known_size == 0:
            return {}
        unique, counts = np.unique(self.known_results, return_counts=True)
        return dict(zip(unique, counts / self.known_size))

    @CacheProperty
    def variance(self) -> float:
        """Calculate the current layer's variance based on known samples"""
        if self.known_size == 0:
            return 0.0

        proportions = self.proportions
        if len(proportions) <= 1:
            return 0.0
        return 1 - sum(p**2 for p in proportions.values())  # 1 - sum(p_i^2)

    @property
    def size(self) -> int:
        """Return the total number of samples in the stratum"""
        return len(self.indices)

    @property
    def known_size(self) -> int:
        """Return the number of known samples in the stratum"""
        return len(self.known_dict)

    @property
    def available_size(self) -> int:
        """Return the number of samples available for sampling in the stratum"""
        return self.size - self.known_size

    @CacheProperty
    def known_indices(self) -> np.ndarray:
        """Get the indices of known samples in this stratum"""
        return np.array(list(self.known_dict.keys()))

    @CacheProperty
    def known_results(self) -> np.ndarray:
        """Get all known sample results for this stratum"""
        return np.array(list(self.known_dict.values()))

    def __repr__(self):
        """Return the basic information of the stratum"""
        return (
            f"Stratum(size={self.size}, "
            f"available={self.available_size}, "
            f"known={self.known_size}, "
            f"variance={self.variance:.3f}, "
            f"tag={self.tag})"
        )

    def _cal_var(self, proportion: float) -> float | None:
        """Calculate this stratum's contribution to the estimated total variance of the specified class"""
        return (
            (self.size**2)
            * ((self.available_size) / (self.size - 1) if self.size > 1 else 1)
            * (proportion * (1 - proportion))
            / self.known_size
        )


class DynamicTextIndex:
    def __init__(
        self,
        texts: list[str],
        n_classes: int = 2,
        n_strata: int | None = None,
        max_strata: int | None = None,
        alpha: float | None = None,
        significance_level: float = 0.95,
        sim_threshold: float = 0.8,
        adjust_threshold: float = 0.45,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        # Create embeddings
        model = SentenceTransformer(
            embedding_model,
            cache_folder="/fs/fast/share/pingtai_cc/models/huggingface/",
            local_files_only=True,
        )
        self.embeddings = model.encode(texts)  # shape: (n_samples, dimension)
        self.dimension = self.embeddings.shape[1]
        if alpha is None:
            alpha = 1 - 20 / len(texts)
        self.alpha = alpha

        # Initialize strata
        if n_strata is None:
            n_strata = min(n_classes * 2, len(texts) // 200)
        # n_strata = 1
        strata_indices = self._init_strata(n_strata)
        self.strata = [
            Stratum(indices, self._get_embedding(indices), self.alpha)
            for indices in strata_indices
        ]

        if max_strata is None:
            max_strata = min(n_classes * 4, len(texts) // 100)
        self.max_strata = max_strata

        for i, stratum in enumerate(self.strata):
            logger.info(f"Stratum {i}: {stratum.size} samples")

        # Initialize parameters
        self.sim_threshold = sim_threshold
        self.adjust_threshold = adjust_threshold
        self.z_value = norm.ppf(1 - significance_level / (2 * n_classes))

        self.known_dict: dict[int, Any] = (
            {}
        )  # Record all known sample results {index: label}

    def sample(self, sample_size: int) -> list[np.ndarray]:
        """Sample from all strata based on optimal allocation strategy"""
        # Calculate initial allocation
        weights = np.array([stratum.available_size for stratum in self.strata])
        allocation_ratios = weights / np.sum(weights)
        stratum_sample_sizes = np.round(sample_size * allocation_ratios).astype(int)

        # Adjust sample sizes to ensure the total equals the target sample size or maximum available samples
        available_size = sum(stratum.available_size for stratum in self.strata)
        max_possible_samples = min(sample_size, available_size)
        stratum_sample_sizes = self._adjust_sample_sizes(
            stratum_sample_sizes, max_possible_samples, weights
        )

        # Execute sampling
        sampled_indices = [
            self.strata[i].sample(size) for i, size in enumerate(stratum_sample_sizes)
        ]

        return sampled_indices

    @invalidate_cache
    def record(self, stratum_id: int, idx: int, result: Any) -> None:
        """Update known sample results"""
        stratum = self.strata[stratum_id]
        self.known_dict[idx] = result
        stratum.record(idx, result)

    @invalidate_cache
    def adjust(self) -> bool:
        """Adjust problematic strata based on all known sample results"""
        for stratum_id, stratum in enumerate(self.strata):
            if stratum.known_size == 0:
                continue
            if self._is_balance(stratum):
                continue

            logger.info(f"\nStratum {stratum_id} needs adjustment: {stratum}\n")
            self._adjust_stratum(stratum_id, stratum)

    @CacheProperty
    def confidence_intervals(
        self,
    ) -> Dict[Any, tuple[float, float]]:
        """Calculate confidence intervals for each class proportion, considering all known sample results"""
        if self.known_size == 0:
            return []

        result = []
        timestamp = datetime.utcnow().isoformat() + "Z"
        for cls, proportion in self.proportions.items():
            total_variance = sum(
                stratum.get_cls_var(cls) / (self.size**2) for stratum in self.strata
            )

            # Calculate confidence intervals
            margin_of_error = self.z_value * np.sqrt(total_variance)
            lower_bound = max(0, proportion - margin_of_error)
            upper_bound = min(1, proportion + margin_of_error)

            # result.append((cls, proportion, (lower_bound, upper_bound)))
            result.append(
                {
                    "class_label": str(cls),
                    "proportion": float(proportion),
                    "confidence_interval": (float(lower_bound), float(upper_bound)),
                    "timestamp": timestamp,
                }
            )
        return result

    @CacheProperty
    def proportions(self) -> dict[Any, float]:
        """Get the proportions of each class"""
        if not self.known_dict:
            return {}
        unique, counts = np.unique(self.known_results, return_counts=True)
        return dict(zip(unique, counts / self.known_size))

    @property
    def n_strata(self) -> int:
        return len(self.strata)

    @property
    def size(self) -> int:
        return len(self.embeddings)

    @property
    def known_size(self) -> int:
        return len(self.known_dict)

    @CacheProperty
    def known_results(self) -> list[Any]:
        return list(self.known_dict.values())

    def __repr__(self) -> str:
        """Return basic information about TextIndex"""
        strata_info = ",\n    ".join(str(stratum) for stratum in self.strata)
        return (
            f"TextIndex(\n"
            f"  total_samples={self.size},\n"
            f"  total_known={self.known_size},\n"
            f"  n_strata={self.n_strata},\n"
            f"  strata=[\n    {strata_info}\n  ],\n"
            f"  sim_threshold={self.sim_threshold},\n"
            f"  adjust_threshold={self.adjust_threshold},\n"
            f"  alpha={self.alpha},\n"
            f"  embedding_dimension={self.dimension}\n"
            f")"
        )

    def _adjust_stratum(self, stratum_id: int, stratum: Stratum):
        """Adjust sample quantities in the stratum"""
        separated_by_class, remained_indices = self._separate_minority_samples(stratum)

        if not separated_by_class:  # If no samples were separated, skip
            logger.debug(f"Stratum {stratum_id}: No samples need to be separated")
            return

        # Update current stratum
        stratum.indices = remained_indices
        stratum.embedding = self._get_embedding(remained_indices)
        for indices in separated_by_class.values():
            self._del_known_result(stratum, indices)

        # Process separated samples
        for cls, indices in separated_by_class.items():
            logger.debug(
                f"Separated {len(indices)} samples of class {cls} from stratum_{stratum_id}"
            )

            if self.n_strata < self.max_strata:
                self._add_stratum(indices)
            else:
                self._merge_strata(stratum_id, indices, cls)

    def _is_balance(self, stratum: Stratum) -> bool:
        """Check if the stratum is balanced"""
        normalized_variance = (
            stratum.variance / (1 - 1 / len(stratum.proportions))
            if len(stratum.proportions) > 1
            else 0
        )
        if normalized_variance <= self.adjust_threshold:
            return True  # If variance is below threshold, no adjustment needed
        return False

    def _adjust_sample_sizes(
        self, sizes: np.ndarray, target_size: int, weights: np.ndarray
    ) -> np.ndarray:
        """Adjust sample size allocation to ensure the total equals the target sample size"""
        curr_total = np.sum(sizes)
        if curr_total == target_size:
            return sizes

        diff = target_size - curr_total
        adjusted_sizes = sizes.copy()

        if diff > 0:
            # Get adjustable stratum indices
            valid_idx = np.where(
                sizes
                < np.array([self.strata[i].available_size for i in range(len(sizes))])
            )[0]
            if len(valid_idx) == 0:
                return adjusted_sizes

            # Start increasing from strata with larger weights
            sorted_idx = valid_idx[np.argsort(weights[valid_idx])[::-1]]
            for i in range(
                len(sorted_idx) * 1000
            ):  # Set an upper limit to prevent infinite loops
                idx = sorted_idx[i % len(sorted_idx)]
                if adjusted_sizes[idx] < self.strata[idx].available_size:
                    adjusted_sizes[idx] += 1
                    diff -= 1
                if diff == 0:
                    break
        else:
            # Get indices of strata with sample size greater than 0
            valid_idx = np.where(sizes > 0)[0]
            if len(valid_idx) == 0:
                return adjusted_sizes

            # Start decreasing from strata with smaller weights
            sorted_idx = valid_idx[np.argsort(weights[valid_idx])]
            for i in range(
                len(sorted_idx) * 1000
            ):  # Set an upper limit to prevent infinite loops
                idx = sorted_idx[i % len(sorted_idx)]
                if adjusted_sizes[idx] > 0:
                    adjusted_sizes[idx] -= 1
                    diff += 1
                if diff == 0:
                    break

        return adjusted_sizes

    def _separate_minority_samples(
        self,
        stratum: Stratum,
    ) -> tuple[dict[Any, np.ndarray], np.ndarray]:
        """Separate minority and majority samples, grouped by class. Uses all known sample results in the stratum."""
        # Group samples by label
        mask = stratum.known_results != stratum.tag
        minority_indices = stratum.known_indices[mask]
        minority_classes = stratum.known_results[mask]

        unique_classes, class_indices = np.unique(minority_classes, return_inverse=True)
        minority_by_class = {
            cls: minority_indices[class_indices == i]
            for i, cls in enumerate(unique_classes)
        }

        remained_indices = np.setdiff1d(stratum.indices, minority_indices)

        # Find similar indices for each class
        for cls, indices in minority_by_class.items():
            similar_indices = self._find_similar_indices(indices, remained_indices)
            minority_by_class[cls] = np.unique(
                np.concatenate([indices, similar_indices])
            )
            # Update remained_indices
            remained_indices = np.setdiff1d(remained_indices, similar_indices)

        return minority_by_class, remained_indices

    def _add_stratum(self, indices_to_add: np.ndarray):
        """添加新的层"""
        new_stratum = Stratum(
            indices_to_add, self._get_embedding(indices_to_add), self.alpha
        )

        self._add_known_result(new_stratum, indices_to_add)
        self.strata.append(new_stratum)
        logger.debug(
            f"Created new stratum: stratum_{self.n_strata-1}(tag={new_stratum.tag}), size={new_stratum.size}"
        )

    def _merge_strata(
        self, stratum_id: int, indices_to_merge: np.ndarray, indices_tag: int
    ):
        """Accelerate stratum merging using Faiss, preferentially merging strata with the same tags"""
        indices_embedding = self._get_embedding(indices_to_merge).reshape(1, -1)

        # Find strata with the same tags
        strata_tags = np.array([s.tag for s in self.strata])
        same_tag_mask = (strata_tags == indices_tag) & (
            np.arange(self.n_strata) != stratum_id
        )
        same_tag_strata = np.where(same_tag_mask)[0]

        if same_tag_strata.size == 0:
            logger.debug(
                f"No stratum with tag={indices_tag} found, adding back to source stratum"
            )
            source_stratum = self.strata[stratum_id]
            source_stratum.indices = np.concatenate(
                [source_stratum.indices, indices_to_merge]
            )
            source_stratum.embedding = self._get_embedding(source_stratum.indices)

            # Update known sample records
            self._add_known_result(source_stratum, indices_to_merge)
            return

        same_tag_embeddings = np.stack(
            [self.strata[i].embedding for i in same_tag_strata]
        )
        faiss.normalize_L2(same_tag_embeddings)
        faiss.normalize_L2(indices_embedding)

        index = faiss.IndexFlatIP(self.dimension)
        index.add(np.ascontiguousarray(same_tag_embeddings))

        _, I = index.search(np.ascontiguousarray(indices_embedding), 1)
        best_stratum_id = same_tag_strata[I[0, 0]]

        # Merge to target stratum
        target_stratum = self.strata[best_stratum_id]
        target_stratum.indices = np.concatenate(
            [target_stratum.indices, indices_to_merge]
        )
        target_stratum.embedding = self._get_embedding(target_stratum.indices)

        # Update known sample records
        self._add_known_result(target_stratum, indices_to_merge)

        logger.debug(
            f"Merged: stratum_{best_stratum_id}(tag={target_stratum.tag}): "
            f"{target_stratum.size - len(indices_to_merge)} += {len(indices_to_merge)} samples"
        )

    def _find_similar_indices(
        self, minority_indices: np.ndarray, majority_indices: np.ndarray
    ) -> np.ndarray:
        """Use Faiss to accelerate similarity search"""
        if len(minority_indices) == 0 or len(majority_indices) == 0:
            return np.array([], dtype=int)

        minority_embeddings = self.embeddings[minority_indices]
        majority_embeddings = self.embeddings[majority_indices]

        # Normalize vectors so the inner product equals cosine similarity
        faiss.normalize_L2(majority_embeddings)
        faiss.normalize_L2(minority_embeddings)

        # Create Faiss index
        index = faiss.IndexFlatIP(self.dimension)
        index.add(np.ascontiguousarray(majority_embeddings))

        D, I = index.search(
            np.ascontiguousarray(minority_embeddings), len(majority_indices)
        )

        # Use numpy boolean indexing
        similar_indices = majority_indices[I[D > self.sim_threshold]]
        return np.unique(similar_indices)

    def _init_strata(self, n_strata: int = 5) -> list[np.ndarray]:
        """Initialize stratification"""
        kmeans = faiss.Kmeans(self.dimension, n_strata, verbose=True)
        embeddings_array = np.ascontiguousarray(self.embeddings.astype("float32"))
        kmeans.train(embeddings_array)
        _, labels = kmeans.index.search(embeddings_array, 1)
        return [np.where(labels.ravel() == i)[0] for i in range(n_strata)]

    def _get_embedding(self, indices: np.ndarray) -> np.ndarray:
        """Calculate the average embedding for the specified sample set"""
        return np.mean(self.embeddings[indices], axis=0)

    def _del_known_result(self, stratum: Stratum, indices: np.ndarray):
        """Delete known sample results"""
        for idx in indices:
            if idx in stratum.known_dict:
                stratum.delete(idx)

    def _add_known_result(self, stratum: Stratum, indices: np.ndarray):
        """Add known sample results"""
        for idx in indices:
            if idx in self.known_dict:
                stratum.record(idx, self.known_dict[idx])
