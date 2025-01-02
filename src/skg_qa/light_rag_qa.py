from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple

RETRIEVAL_QUERY = """
MATCH (node)-[r]-(neighbor)
RETURN 
    node.description AS text,
    score,
    {nodeId: node.id, nodeDescription: node.description, nodeLabels: labels(node), neighborId: neighbor.id, neighborDescription: neighbor.description, neighborLabels: labels(neighbor), sourceId: startNode(r).id, targetId: endNode(r).id, relationshipType: type(r)} AS metadata
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

class LightRagQaLLM():
    def __init__(self, graph:Neo4jGraph, config:dict, exclude:List[str]=["Chunk"]):
        self.stores = _build_embedding_index(graph, config['neo4j_uri'], config['neo4j_username'], config['neo4j_password'], exclude, RETRIEVAL_QUERY)
        self.client = OpenAI(api_key=config['openai_api_key'], base_url=config['openai_base_url'])
    
    def build_context(self, query:str, topk:int=30) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        retrieve_docs = []
        for store in tqdm(self.stores):
            retrieve_docs += store.similarity_search_with_score(query=query, k=10)
        retrieve_docs.sort(key=lambda x: x[1], reverse=True)
        retrieve_docs = retrieve_docs[:topk]
        
        retrieve_docs = [docs[0].metadata for docs in retrieve_docs]
        
        all_nodes_info = set()
        all_relationships_info = set()

        for doc in retrieve_docs:
            node_info = (doc.get('nodeId', 'UNKNOWN'), doc.get('nodeDescription', 'UNKNOWN'), doc.get('nodeLabels', 'UNKNOWN')[0])
            neighbor_info = (doc.get('neighborId', 'UNKNOWN'), doc.get('neighborDescription', 'UNKNOWN'), doc.get('neighborLabels', 'UNKNOWN')[0])
            all_nodes_info.update([node_info, neighbor_info])
            
            all_relationships_info.update([(doc.get('sourceId', 'UNKNOWN'), doc.get('relationshipType', 'UNKNOWN'), doc.get('targetId', 'UNKNOWN'))])

        all_nodes_info = list(all_nodes_info)
        all_relationships_info = list(all_relationships_info)
        
        all_nodes_info = [[idx] + list(info) for idx, info in enumerate(all_nodes_info)]
        all_relationships_info = [[idx] + list(info) for idx, info in enumerate(all_relationships_info)]

        nodes_df = pd.DataFrame(all_nodes_info, columns=["id", "NodeId", "NodeDescription", "NodeType"])
        relationships_df = pd.DataFrame(all_relationships_info, columns=["id", "SourceNode", "Relationship", "TargetNode"])
        
        all_nodes_info_head = [["id", "NodeId", "NodeDescription", "NodeType"]] + all_nodes_info
        all_relationships_info_head = [["id", "SourceNode", "Relationship", "TargetNode"]] + all_relationships_info
        
        local_query_context = f"""
        -----Entities-----
        ```csv
        {all_nodes_info_head}
        ```
        -----Relationships-----
        ```csv
        {all_relationships_info_head}
        ```
        """
        
        return local_query_context, nodes_df, relationships_df
    
    def invoke(self, query:str, topk:int=30) -> dict:
        context, nodes_df, relationships_df = self.build_context(query=query, topk=topk)
        
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
        
        return {"context": (nodes_df, relationships_df), "result": response.choices[0].message.content}