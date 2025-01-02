"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

import argparse

import torch
import json
from sentence_transformers import SentenceTransformer
import sentence_transformers.util as util
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Semantic search with SentenceTransformer")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="autodl-tmp/models_during_training/all-roberta-large-v1/checkpoint-698",
        help="Path to the SentenceTransformer model"
    )
    parser.add_argument(
        "--corpus_path", 
        type=str, 
        default="data/test_corpus.json",
        help="Path to the corpus JSON file"
    )
    parser.add_argument(
        "--benchmark_path", 
        type=str, 
        default="data/test_benchmark_addids.json",
        help="Path to the benchmark JSON file"
    )
    parser.add_argument(
        "--n", 
        type=int, 
        default=1,
        help="Number of top results to consider (n)"
    )

    args = parser.parse_args()

    # Load model
    embedder = SentenceTransformer(args.model_path)

    # Load corpus
    with open(args.corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Encode corpus
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Load benchmark
    with open(args.benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Filter queries and answer IDs based on item_count
    n = args.n
    queries = [
        list(item['survey'].values())[0] 
        for item in benchmark 
        if n < item['item_count'] < 200
    ]
    answer_ids = [
        item['citations_ids'] 
        for item in benchmark 
        if n < item['item_count'] < 200
    ]

    # Encode queries
    query_embeddings = embedder.encode(queries, convert_to_tensor=True)

    # Perform semantic search
    results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=n)

    # Extract search IDs
    search_ids = [[hit['corpus_id'] for hit in result] for result in results]

    # Calculate accuracy
    accuracy = []
    for ans, search in zip(answer_ids, search_ids):
        common_count = len(set(ans) & set(search))
        accuracy.append(common_count / n)

    # Print metrics
    print(f"mean of accuracy @{n} = {np.mean(accuracy)}")
    print(f"25 quantile of accuracy @{n} = {np.quantile(accuracy, 0.25)}")
    print(f"50 quantile of accuracy @{n} = {np.quantile(accuracy, 0.50)}")
    print(f"75 quantile of accuracy @{n} = {np.quantile(accuracy, 0.75)}")

if __name__ == "__main__":
    main()
