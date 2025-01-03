
import argparse
from tqdm import tqdm

import torch
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import sentence_transformers.util as util
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

from llm_search import llm_search

def split_text(text, max_length=2000):
    # 初始化分句器
    sentences = sent_tokenize(text)
    
    # 存储最终的分割结果
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果当前句子长度加上当前内容不超过最大限制，添加到当前片段
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            # 如果当前句子加上已有内容超出最大限制，保存当前片段并开始新的片段
            if current_chunk:  # 确保当前片段不为空
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    # 添加最后一个片段
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

prompt_template = """
#### Instruction ####
You are tasked with identifying {top_k} abstracts from a list of 50, which are **most likely** to be referenced or included in the survey paper. Use the survey abstract as the reference point for your analysis. Return the results strictly in the following JSON format:
{{
  "idx_of_correct_abstract": ["","",...,""]
}}

#### Output Requirements ####
- Put the list index of the correct abstract in idx_of_correct_abstract, the list index should be ranging from 0 to 49
- Ensure no additional output other than the required JSON.
- Ensure the "idx_of_correct_abstract" contains exactly {top_k} indexes, even if more abstract seem relevant.

#### Input Details ####
- **Survey Abstract**: The main abstract of the survey paper that serves as the selection criteria.
- **List of Abstracts**: A list of 50 abstracts to evaluate against the survey abstract.

#### Survey Abstract ####
{survey_abs}

#### List of Abstracts ####
{paper_abs}
"""

# Load model
biencoder = SentenceTransformer("all-roberta-large-v1")
# # crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())
# crossencoder = CrossEncoder("cross-encoder/stsb-roberta-large")

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
results_for_llm = [[split_text(corpus[id])[0] for id in ids] for q, ids in zip(queries,search_ids)]

top_k = 30
llm_search_ids = []
llm_path = "gpt-4o"
for q, list_of_abs, ids in tqdm(zip(queries, results_for_llm, search_ids), total=len(queries), desc="Processing queries"):
    indices = llm_search(q, list_of_abs, top_k, prompt_template, llm_path)
    search_ids_2 = [ids[int(idx)] for idx in indices]
    llm_search_ids.append(search_ids_2)

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

# Calculate biencoder + llm reranker accuracy
accuracy = []
for ans, search in zip(answer_ids, llm_search_ids):
    common_count = len(set(ans) & set(search))
    accuracy.append(common_count / n)

# Print metrics
print("below is results of biencoder + llm reranker")
print(f"mean of accuracy @{top_k} = {np.mean(accuracy)}")
print(f"25 quantile of accuracy @{top_k} = {np.quantile(accuracy, 0.25)}")
print(f"50 quantile of accuracy @{top_k} = {np.quantile(accuracy, 0.50)}")
print(f"75 quantile of accuracy @{top_k} = {np.quantile(accuracy, 0.75)}")


