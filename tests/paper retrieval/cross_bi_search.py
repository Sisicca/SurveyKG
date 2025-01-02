"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

import argparse

import torch
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import sentence_transformers.util as util
import numpy as np

# Load model
biencoder = SentenceTransformer("all-roberta-large-v1")
# crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())
crossencoder = CrossEncoder("cross-encoder/stsb-roberta-large")

# Load corpus
with open("data/test_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# Encode corpus
corpus_embeddings = biencoder.encode(corpus, convert_to_tensor=True)

# Load benchmark
with open("data/test_benchmark_addids.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Filter queries and answer IDs based on item_count
n = 50
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
query_embeddings = biencoder.encode(queries, convert_to_tensor=True)

# Perform semantic search
results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=n)

# Extract search IDs
search_ids = [[hit['corpus_id'] for hit in result] for result in results]
results_for_crossencoder = [[(q,corpus[id]) for id in ids] for q, ids in zip(queries,search_ids)]

top_k = 30
cross_search_ids = []
for list_of_pairs, ids in zip(results_for_crossencoder, search_ids):
    crossencoder_scores = crossencoder.predict(list_of_pairs)
    crossencoder_scores_tensor = torch.tensor(crossencoder_scores) 
    scores, indices = torch.topk(crossencoder_scores_tensor, k=top_k)
    search_ids_2 = [ids[idx] for idx in indices]
    cross_search_ids.append(search_ids_2)

# Calculate biencoder accuracy
accuracy = []
for ans, search in zip(answer_ids, search_ids):
    common_count = len(set(ans) & set(search))
    accuracy.append(common_count / n)

# Print metrics
print("below is results of biencoder")
print(f"mean of accuracy @{n} = {np.mean(accuracy)}")
print(f"25 quantile of accuracy @{n} = {np.quantile(accuracy, 0.25)}")
print(f"50 quantile of accuracy @{n} = {np.quantile(accuracy, 0.50)}")
print(f"75 quantile of accuracy @{n} = {np.quantile(accuracy, 0.75)}")

# Calculate biencoder + crossencoder accuracy
accuracy = []
for ans, search in zip(answer_ids, cross_search_ids):
    common_count = len(set(ans) & set(search))
    accuracy.append(common_count / n)

# Print metrics
print("below is results of biencoder + crossencoder")
print(f"mean of accuracy @{top_k} = {np.mean(accuracy)}")
print(f"25 quantile of accuracy @{top_k} = {np.quantile(accuracy, 0.25)}")
print(f"50 quantile of accuracy @{top_k} = {np.quantile(accuracy, 0.50)}")
print(f"75 quantile of accuracy @{top_k} = {np.quantile(accuracy, 0.75)}")


