from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def dirichlet_noniid_split(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
) -> Dict[int, List[int]]:
    """Class-wise Dirichlet allocation (non-IID label skew)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for i, part in enumerate(splits):
            client_indices[i].extend(part.tolist())

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def pathological_shard_split(
    labels: Sequence[int],
    num_clients: int,
    seed: int,
    shards_per_client: int = 2,
) -> Dict[int, List[int]]:
    """Pathological non-IID split via label-sorted shards.

    This follows the standard shard-based construction used in classical FL
    baselines: sort samples by label, cut the ordered list into
    ``num_clients * shards_per_client`` contiguous shards, then assign
    ``shards_per_client`` shards to each client.
    """
    if shards_per_client < 1:
        raise ValueError(f"shards_per_client must be >= 1, got {shards_per_client}")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_samples = int(labels.shape[0])
    num_shards = num_clients * shards_per_client
    if num_shards > num_samples:
        raise ValueError(
            f"pathological split requires at least {num_shards} samples, got {num_samples}"
        )

    # Shuffle before sorting so same-label samples do not preserve any
    # dataset-order artifact inside each shard.
    shuffled = rng.permutation(num_samples)
    sorted_indices = shuffled[np.argsort(labels[shuffled], kind="stable")]
    shards = [shard.tolist() for shard in np.array_split(sorted_indices, num_shards)]

    shard_order = rng.permutation(num_shards)
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    for client_id in range(num_clients):
        take = shard_order[
            client_id * shards_per_client : (client_id + 1) * shards_per_client
        ]
        for shard_id in take:
            client_indices[client_id].extend(shards[int(shard_id)])
        rng.shuffle(client_indices[client_id])

    return client_indices


def build_federated_split(
    labels: Sequence[int],
    num_clients: int,
    partition_method: str,
    alpha: float,
    seed: int,
    patho_shards_per_client: int = 2,
) -> Dict[int, List[int]]:
    if partition_method == "dirichlet":
        return dirichlet_noniid_split(labels, num_clients, alpha, seed)
    if partition_method == "patho":
        return pathological_shard_split(
            labels,
            num_clients,
            seed,
            shards_per_client=patho_shards_per_client,
        )
    raise ValueError(f"Unknown partition_method: {partition_method}")
