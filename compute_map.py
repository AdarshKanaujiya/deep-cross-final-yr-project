"""
compute_map.py

Mean Average Precision (mAP) & Recall@K computation for multi-label
satellite image retrieval on BigEarthNet.

A retrieved image is considered relevant when it shares at least one
land-cover label with the query image.
"""

import re
import numpy as np
import pandas as pd


# ============================================================
# LABEL PARSING
# ============================================================

def parse_labels(label_input):
    """
    Convert the label format used in BigEarthNet CSV files into
    a Python list of label strings.

    Example input:
        "['Pastures' 'Urban fabric']"

    Output:
        ['Pastures', 'Urban fabric']
    """
    if isinstance(label_input, (list, set, tuple)):
        return [str(item).strip() for item in label_input if str(item).strip()]

    if not isinstance(label_input, str):
        return []

    # Extract text enclosed inside single quotes (handles space/newline separators)
    labels = re.findall(r"'([^']*)'", label_input)
    if not labels:
        # Fallback if double quotes or unquoted single label
        labels = re.findall(r'"([^"]*)"', label_input)
    if not labels:
        # Simple string fallback
        clean_str = label_input.strip("[]'\" ")
        if clean_str:
            labels = [clean_str]

    return [label.strip() for label in labels if label.strip()]


# ============================================================
# CHECK LABEL OVERLAP
# ============================================================

def labels_overlap(query_labels, retrieved_labels):
    """
    Check whether the query and retrieved image
    have at least one common land-cover label.

    Returns:
        True  -> relevant
        False -> not relevant
    """
    q_set = set(parse_labels(query_labels))
    r_set = set(parse_labels(retrieved_labels))
    return len(q_set.intersection(r_set)) > 0


# ============================================================
# AVERAGE PRECISION FOR ONE QUERY
# ============================================================

def average_precision(query_labels, retrieved_results):
    """
    Calculate Average Precision (AP) for one query.

    Parameters
    ----------
    query_labels : list or str
        Ground-truth labels of the query image.

    retrieved_results : list
        Retrieved results in ranked order (each element is label list/str).

    Returns
    -------
    float
        Average Precision for this query.
    """
    q_labels = parse_labels(query_labels)
    if not q_labels or not retrieved_results:
        return 0.0

    number_of_relevant = 0
    precision_sum = 0.0

    for rank, r_labels in enumerate(retrieved_results, start=1):
        relevant = labels_overlap(q_labels, r_labels)
        if relevant:
            number_of_relevant += 1
            precision_at_rank = number_of_relevant / rank
            precision_sum += precision_at_rank

    if number_of_relevant == 0:
        return 0.0

    return precision_sum / number_of_relevant


# ============================================================
# MEAN AVERAGE PRECISION (mAP)
# ============================================================

def mean_average_precision(all_queries):
    """
    Calculate mean Average Precision (mAP) across multiple queries.

    Parameters
    ----------
    all_queries : list of dictionaries
        Each dict has:
        {
            "query_labels": ['Forest', 'Grassland'],
            "retrieved": [['Forest'], ['Urban fabric'], ...]
        }

    Returns
    -------
    float
        mAP score.
    """
    average_precisions = []

    for query in all_queries:
        ap = average_precision(query["query_labels"], query["retrieved"])
        average_precisions.append(ap)

    if not average_precisions:
        return 0.0

    return float(np.mean(average_precisions))


# ============================================================
# FULL EVALUATION FUNCTION (mAP & RECALL@K)
# ============================================================

def evaluate_retrieval_metrics(query_labels_list, gallery_labels_list, retrieved_indices_matrix, k_values=[1, 5, 10]):
    """
    Evaluates retrieval performance across all queries.

    query_labels_list: list of label strings/lists for each query
    gallery_labels_list: list of label strings/lists for each gallery item
    retrieved_indices_matrix: 2D array (N_queries, K) containing indices into gallery
    k_values: list of depths (e.g. [1, 5, 10])

    Returns: dict containing mAP@10, Recall@1, Recall@5, Recall@10, etc.
    """
    num_queries = len(query_labels_list)
    results = {f"Recall@{k}": 0.0 for k in k_values}
    results["mAP@10"] = 0.0

    ap_list = []

    for i in range(num_queries):
        q_labels = parse_labels(query_labels_list[i])
        retrieved_idx = retrieved_indices_matrix[i]
        retrieved_labels_for_q = [parse_labels(gallery_labels_list[idx]) for idx in retrieved_idx]

        # Compute AP@10
        ap = average_precision(q_labels, retrieved_labels_for_q[:10])
        ap_list.append(ap)

        # Compute Recall@K (1 if at least one relevant item in top-K)
        for k in k_values:
            top_k_retrieved = retrieved_labels_for_q[:k]
            if any(labels_overlap(q_labels, r_l) for r_l in top_k_retrieved):
                results[f"Recall@{k}"] += 1.0

    return results


def evaluate_retrieval_by_patch_id(metadata_df, query_patch_ids, retrieved_patch_ids_matrix, k_values=[1, 5, 10]):
    """
    Evaluates retrieval performance when query and gallery items are represented by patch_id strings.
    Looks up labels from metadata_df using patch_id.

    metadata_df: pandas DataFrame containing 'patch_id' and 'labels' columns
    query_patch_ids: list of query patch_id strings (N_queries,)
    retrieved_patch_ids_matrix: 2D list/array of retrieved patch_id strings (N_queries, K)
    k_values: list of depths (e.g. [1, 5, 10])

    Returns: dict containing mAP@10, Recall@1, Recall@5, Recall@10
    """
    patch_to_labels = dict(zip(metadata_df["patch_id"], metadata_df["labels"].apply(parse_labels)))

    query_labels = [patch_to_labels.get(pid, []) for pid in query_patch_ids]
    gallery_labels_matrix = [
        [patch_to_labels.get(r_pid, []) for r_pid in retrieved_row]
        for retrieved_row in retrieved_patch_ids_matrix
    ]

    num_queries = len(query_patch_ids)
    results = {f"Recall@{k}": 0.0 for k in k_values}
    ap_list = []

    for i in range(num_queries):
        q_l = query_labels[i]
        r_l_list = gallery_labels_matrix[i]

        ap = average_precision(q_l, r_l_list[:10])
        ap_list.append(ap)

        for k in k_values:
            top_k = r_l_list[:k]
            if any(labels_overlap(q_l, r_l) for r_l in top_k):
                results[f"Recall@{k}"] += 1.0

    results["mAP@10"] = float(np.mean(ap_list)) if ap_list else 0.0
    for k in k_values:
        results[f"Recall@{k}"] = float(results[f"Recall@{k}"] / num_queries) if num_queries > 0 else 0.0

    return results


# ============================================================
# SELF TEST SUITE
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing compute_map.py implementation")
    print("=" * 60)

    # Test 1: Label parsing
    example_raw = "['Pastures' 'Urban fabric']"
    parsed = parse_labels(example_raw)
    print("\nTest 1 - Label parsing")
    print("Input :", example_raw)
    print("Output:", parsed)
    assert parsed == ["Pastures", "Urban fabric"], "Parse test failed!"

    # Test 2: Label overlap
    q = ["Forest", "Grassland"]
    r = ["Forest", "Water"]
    overlap = labels_overlap(q, r)
    print("\nTest 2 - Label overlap")
    print("Query    :", q)
    print("Retrieved:", r)
    print("Relevant :", overlap)
    assert overlap is True, "Overlap test failed!"

    # Test 3: Average Precision
    query_labels = ["Forest", "Grassland"]
    retrieved_results = [
        ["Forest"],
        ["Urban fabric"],
        ["Grassland", "Forest"],
        ["Water"],
        ["Pastures"]
    ]
    ap = average_precision(query_labels, retrieved_results)
    print("\nTest 3 - Average Precision")
    print("Query labels:", query_labels)
    print(f"AP = {ap:.4f}")
    assert ap > 0, "AP test failed!"

    # Test 4: Mean Average Precision
    all_queries = [
        {
            "query_labels": ["Forest", "Grassland"],
            "retrieved": [["Forest"], ["Urban fabric"], ["Grassland", "Forest"], ["Water"], ["Pastures"]]
        },
        {
            "query_labels": ["Urban fabric"],
            "retrieved": [["Forest"], ["Urban fabric"], ["Water"], ["Urban fabric"], ["Grassland"]]
        },
        {
            "query_labels": ["Water"],
            "retrieved": [["Forest"], ["Grassland"], ["Water"], ["Urban fabric"], ["Water"]]
        }
    ]
    map_score = mean_average_precision(all_queries)
    print("\nTest 4 - Mean Average Precision")
    print(f"mAP = {map_score:.4f}")
    assert map_score > 0, "mAP test failed!"

    print("\n" + "=" * 60)
    print("mAP implementation test COMPLETED SUCCESSFULLY ✅")
    print("=" * 60)
