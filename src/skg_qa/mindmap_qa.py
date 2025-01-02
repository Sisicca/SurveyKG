from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import os
from typing import Any, Dict, List, Optional, Union, Tuple


SEMANTIC_RETRIEVAL_QUERY = """
RETURN 
    node.description AS text,
    score,
    {node_index: id(node)} AS metadata
"""

NDNW_ST_QUERY = """
// 收集所有终端节点
MATCH (t)
WHERE id(t) IN {terminal_node_ids}
WITH collect(t) AS terminals

// 生成所有终端节点对
UNWIND range(0, size(terminals)-2) AS i
UNWIND range(i+1, size(terminals)-1) AS j
WITH terminals[i] AS t1, terminals[j] AS t2

// 查找每对终端节点之间的最短路径（无向）
MATCH path = shortestPath((t1)-[*..{max_k_hop}]-(t2))
WITH collect(path) AS paths

// 提取所有路径中的关系，去除重复
UNWIND paths AS p
UNWIND relationships(p) AS rel
// RETURN DISTINCT rel
// RETURN p
RETURN 
    {{start_node_id: startNode(rel).id, start_node_desc: startNode(rel).description, start_node_type: labels(startNode(rel)), rel_type: type(rel), end_node_id: endNode(rel).id, end_node_desc: endNode(rel).description, end_node_type: labels(endNode(rel))}} AS metadata
"""

NEIGHBOR_QUERY = """
MATCH (node)-[r]-(neighbor)
WHERE id(node) IN {terminal_node_ids}
// RETURN node, r, neighbor
RETURN 
    {{nodeId: node.id, nodeDescription: node.description, nodeLabels: labels(node), neighborId: neighbor.id, neighborDescription: neighbor.description, neighborLabels: labels(neighbor), sourceId: startNode(r).id, targetId: endNode(r).id, relationshipType: type(r)}} AS metadata
"""

def _build_embedding_index(graph:Neo4jGraph, neo4j_url:str, neo4j_username:str, neo4j_pwd:str, exclude:List[str]=["Chunk"], retrieval_query:str="") -> List[Neo4jVector]:
    all_types = graph.query(
                    """
                    MATCH (n)
                    RETURN DISTINCT labels(n) AS label
                    """
                )
    all_types = [t["label"][0] for t in all_types if t["label"][0] not in exclude]
    all_types_store = []
    for t in tqdm(all_types):
        all_types_store.append(Neo4jVector.from_existing_graph(
                                embedding=OpenAIEmbeddings(),
                                url=neo4j_url,
                                username=neo4j_username,
                                password=neo4j_pwd,
                                index_name=f"{t.lower()}_index",
                                node_label=t,
                                text_node_properties=["description"],
                                embedding_node_property="embedding",
                                retrieval_query=retrieval_query
                                ))
    return all_types_store

class MindMapQaLLM():
    def __init__(self, graph:Neo4jGraph, config:dict, exclude:List[str]=["Chunk"]):
        self.graph = graph
        self.stores = _build_embedding_index(graph, config['neo4j_uri'], config['neo4j_username'], config['neo4j_password'], exclude, SEMANTIC_RETRIEVAL_QUERY)
        self.client = OpenAI(api_key=config['openai_api_key'], base_url=config['openai_base_url'])
        
    def build_context(self, query:str, max_terminal:int=3, max_k_hop:int=5) -> Tuple[str,pd.DataFrame,pd.DataFrame,Tuple[List[dict],List[dict]]]:
        def get_element(metadata, element):
            return metadata.get(element, "UNKOWN") if metadata.get(element, "UNKOWN") else ""
        
        retrieve_docs = []
        for store in tqdm(self.stores):
            retrieve_docs += store.similarity_search_with_score(query=query, k=max_terminal)
        
        retrieve_node_ids = []
        for doc in retrieve_docs:
            node_index = doc[0].metadata['node_index']
            score = doc[1]
            retrieve_node_ids.append(
                (node_index, score)
            )
        retrieve_node_ids = sorted(retrieve_node_ids, key=lambda x: x[1], reverse=True)
        retrieve_node_ids = retrieve_node_ids[:max_terminal]
        terminal_node_ids = [item [0] for item in retrieve_node_ids]
        
        # Steiner Tree Subgraph
        st_results = self.graph.query(NDNW_ST_QUERY.format(terminal_node_ids=terminal_node_ids, max_k_hop=max_k_hop))
        while not st_results:
            max_k_hop += 1
            st_results = self.graph.query(NDNW_ST_QUERY.format(terminal_node_ids=terminal_node_ids, max_k_hop=max_k_hop))
        
        st_nodes_info = set()
        st_rels_info = set()

        for result in st_results:
            metadata = result["metadata"]
            
            start_info = (get_element(metadata, 'start_node_id'), get_element(metadata, 'start_node_desc'), get_element(metadata, 'start_node_type')[0])
            end_info = (get_element(metadata, 'end_node_id'), get_element(metadata, 'end_node_desc'), get_element(metadata, 'end_node_type')[0])
            
            st_nodes_info.update([start_info, end_info])
            st_rels_info.update([(get_element(metadata, 'start_node_id'), get_element(metadata, 'rel_type'), get_element(metadata, 'end_node_id'))])

        st_nodes_info = list(st_nodes_info)
        st_rels_info = list(st_rels_info)
        
        # Neighbor Subgraph
        neighbor_results = self.graph.query(NEIGHBOR_QUERY.format(terminal_node_ids=terminal_node_ids))
        
        neighbor_nodes_info = set()
        neighbor_rels_info = set()

        for result in neighbor_results:
            metadata = result["metadata"]
            
            node_info = (get_element(metadata, 'nodeId'), get_element(metadata, 'nodeDescription'), get_element(metadata, 'nodeLabels')[0])
            neighbor_info = (get_element(metadata, 'neighborId'), get_element(metadata, 'neighborDescription'), get_element(metadata, 'neighborLabels')[0])
            
            neighbor_nodes_info.update([node_info, neighbor_info])
            neighbor_rels_info.update([(get_element(metadata, 'sourceId'), get_element(metadata, 'relationshipType'), get_element(metadata, 'targetId'))])
            
        neighbor_nodes_info = list(neighbor_nodes_info)
        neighbor_rels_info = list(neighbor_rels_info)

        # 聚合 Steiner Tree Subgraph 和 Neighbor Subgraph
        all_nodes_info = list(set(st_nodes_info + neighbor_nodes_info))
        all_rels_info = list(set(st_rels_info+neighbor_rels_info))

        all_nodes_info_idx = [[idx] + list(info) for idx, info in enumerate(all_nodes_info)]
        all_rels_info_idx = [[idx] + list(info) for idx, info in enumerate(all_rels_info)]

        all_nodes_df = pd.DataFrame(all_nodes_info_idx, columns=["id", "NodeId", "NodeDescription", "NodeType"])
        all_rels_df = pd.DataFrame(all_rels_info_idx, columns=["id", "SourceNode", "Relationship", "TargetNode"])

        all_nodes_info_head = [["id", "NodeId", "NodeDescription", "NodeType"]] + all_nodes_info_idx
        all_rels_info_head = [["id", "SourceNode", "Relationship", "TargetNode"]] + all_rels_info_idx

        subgraph_query_context = f"""
        -----Entities-----
        ```csv
        {all_nodes_info_head}
        ```
        -----Relationships-----
        ```csv
        {all_rels_info_head}
        ```
        """
        
        # 生成 subgraph html
        html_nodes = []
        html_relationships = []

        node_idx_record = {}

        for node_info in all_nodes_info_idx:
            node_idx, node_overview, node_desc, node_type = node_info
            html_nodes.append(
                {
                    'id': node_idx,
                    'overview': node_overview,
                    'label': node_overview[:5],
                    'type': node_type,
                    'description': node_desc
                }
            )
            node_idx_record[node_overview] = node_idx

        for rel_info in all_rels_info_idx:
            _, source_node_overview, rel_type, target_node_overview = rel_info
            source_node_idx = node_idx_record[source_node_overview]
            target_node_idx = node_idx_record[target_node_overview]
            
            html_relationships.append(
                {
                    'source': source_node_idx,
                    'target': target_node_idx,
                    'label': rel_type
                }
            )
        
        return subgraph_query_context, all_nodes_df, all_rels_df, (html_nodes, html_relationships)
    
    def invoke(self, query:str, max_terminal:int=3, max_k_hop:int=5) -> dict:
        context, nodes_df, relationships_df, subgraph = self.build_context(query=query, max_terminal=max_terminal, max_k_hop=max_k_hop)
        
        system_prompt = f"""---角色---
        
        你是一名帮助回答关于所提供表格数据问题的助手。
        
        
        ---目标---
        
        生成一个高质量的响应，回答用户的问题，总结输入数据表中的所有信息，并结合任何相关的常识。 如果你不知道答案，就直接说不知道。不要编造任何内容。 不要包含没有提供支持证据的信息。
        
        ---数据表--
        
        {context}
        
        这个提示的目的是指导AI生成一个结构化的、基于数据表的响应，同时确保信息的准确性和可靠性。
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        
        return {"context": (nodes_df, relationships_df), "result": response.choices[0].message.content, "subgraph": subgraph}