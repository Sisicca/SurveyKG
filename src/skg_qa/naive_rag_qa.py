from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from tqdm import tqdm
from openai import OpenAI
from typing import List

def _build_embedding_index(graph:Neo4jGraph, neo4j_uri, neo4j_username, neo4j_pwd, exclude=["Chunk"], retrieval_query="") -> List[Neo4jVector]:
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
                                url=neo4j_uri,
                                username=neo4j_username,
                                password=neo4j_pwd,
                                index_name=f"{t.lower()}_index",
                                node_label=t,
                                text_node_properties=["description"],
                                embedding_node_property="embedding",
                                retrieval_query=retrieval_query
                                ))
    return all_types_store
    
class RagQaLLM():
    def __init__(self, graph:Neo4jGraph, config:dict, exclude=["Chunk"]):
        self.stores = _build_embedding_index(graph, config['neo4j_uri'], config['neo4j_username'], config['neo4j_password'], exclude)
        self.client = OpenAI(api_key=config['openai_api_key'], base_url=config['openai_base_url'])
        
    def search_docs(self, query:str, topk:int=15) -> List[str]:
        retrieve_docs = []
        for store in tqdm(self.stores):
            retrieve_docs += store.similarity_search_with_score(query=query, k=15)
        retrieve_docs.sort(key=lambda x: x[1], reverse=True)
        return retrieve_docs[:topk]
    
    def invoke(self, query:str, topk:int=15) -> dict:
        docs = self.search_docs(query=query, topk=topk)
        docs = [doc[0].page_content.replace("\ndescription: ", "") for doc in docs]
        
        def pretty_print(docs:List[str]) -> str:
            pretty_format = ""
            for idx, doc in enumerate(docs):
                pretty_format += f"Document{idx+1}:\n"
                pretty_format += doc + "\n" + "-"*100 + "\n"
            return pretty_format
        
        docs_merge = pretty_print(docs)
        
        prompt_template = """
        你是一个知识渊博的助手。根据以下文档内容回答问题。
        如果你不知道答案，就直接说不知道。不要编造任何内容。 不要包含没有提供支持证据的信息。
        使答案听起来像是对问题的回应，而不提及你是基于给定信息得出的结果。
        
        文档内容：
        {documents}

        问题：
        {question}

        请根据文档内容提供详细的回答。
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt_template.format(documents=docs_merge, question=query)}
            ]
        )
        
        return {"documents": docs_merge, "result": response.choices[0].message.content}