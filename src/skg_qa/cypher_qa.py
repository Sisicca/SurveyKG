from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain, extract_cypher, get_function_response, INTERMEDIATE_STEPS_KEY
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from typing import Any, Dict, List, Optional, Union
import os

EXAMPLES = [
    {
        "question": "知识图谱补全与建模这个方向下有多少论文？",
        "query": """
        MATCH (p:Paper)-[:RESEARCH]->(:Problem)<-[:INCLUDE]-(sd:Secondary_Direction)
        WHERE sd.id = '知识图谱补全与建模'
        RETURN count(DISTINCT p) AS paperCount
        UNION
        MATCH (p:Paper)-[:RESEARCH]->(:Problem)<-[:INCLUDE]-(:Secondary_Direction)<-[:INCLUDE]-(pd:Primary_Direction)
        WHERE pd.id = '知识图谱补全与建模'
        RETURN count(DISTINCT p) AS paperCount
        """
    },
    {
        "question": "哪些方向研究了知识图谱辅助大模型推理？",
        "query": """
        MATCH (pd:Primary_Direction)-[:INCLUDE]->(:Secondary_Direction)-[:INCLUDE]->(p:Problem)
        WHERE p.description CONTAINS "知识图谱辅助" OR p.description CONTAINS "大模型推理" OR p.description CONTAINS "知识图谱推理"
        RETURN DISTINCT pd.id
        """
    },
    {
        "question": "该领域叫什么？都有哪些研究方向？",
        "query": """
        MATCH (d:Domain)-[:INCLUDE]->(pd:Primary_Direction)-[:INCLUDE]->(sd:Secondary_Direction)
        RETURN d.id AS Domain, 
            collect(DISTINCT pd.id) AS Primary_Directions, 
            collect(DISTINCT sd.id) AS Secondary_Directions
        """
    },
    {
        "question": "最热门的研究方向是什么？",
        "query": """
        MATCH (d:Domain)-[:INCLUDE]->(pd:Primary_Direction)-[:INCLUDE]->(sd:Secondary_Direction)-[:INCLUDE]->(p:Problem)<-[:RESEARCH]-(paper:Paper)
        RETURN pd.id AS Primary_Direction, COUNT(paper) AS PaperCount
        ORDER BY PaperCount DESC
        LIMIT 1
        """
    },
    {
        "question": "“知识图谱与语言模型的结合”这个研究方向有哪些论文？",
        "query": """
        MATCH (p:Paper)-[:RESEARCH]->(:Problem)<-[:INCLUDE]-(sd:Secondary_Direction)
        WHERE sd.id = '知识图谱与语言模型的结合'
        RETURN p
        UNION
        MATCH (p:Paper)-[:RESEARCH]->(:Problem)<-[:INCLUDE]-(:Secondary_Direction)<-[:INCLUDE]-(pd:Primary_Direction)
        WHERE pd.id = '知识图谱与语言模型的结合'
        RETURN p
        """
    },
    {
        "question": "“知识图谱与语言模型的结合”这个研究方向会用到哪些技术？",
        "query": """
        MATCH (d:Primary_Direction)-[:INCLUDE]->(pd:Secondary_Direction)-[:INCLUDE]->(:Problem)<-[:RESEARCH]-(:Paper)-[:USE]->(m:Method)<-[:INCLUDE]-(t:Technology)
        WHERE d.id = "知识图谱与语言模型的结合" OR pd.id = "知识图谱与语言模型的结合"
        RETURN DISTINCT t.id
        """
    },
    {
        "question": "有哪些方法经常被使用？",
        "query": """
        MATCH (t:Technology)-[:INCLUDE]->(m:Method)<-[:USE]-(p:Paper)
        RETURN t.id, COUNT(p) AS usage_count
        ORDER BY usage_count DESC
        """
    }  
]

class CustomGraphCypherQAChain(GraphCypherQAChain):
    """Custom chain for question-answering against a graph with additional functionality."""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        args = {
            "question": question,
            "schema": self.graph_schema,
        }
        args.update(inputs)

        intermediate_steps: List = []

        generated_cypher = self.cypher_generation_chain.run(args, callbacks=callbacks)

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)

        # Correct Cypher query if enabled
        if self.cypher_query_corrector:
            generated_cypher = self.cypher_query_corrector(generated_cypher)

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_cypher})

        # Retrieve and limit the number of results
        if generated_cypher:
            context = self.graph.query(generated_cypher)[: self.top_k]
        else:
            context = []

        if self.return_direct:
            final_result = context
        else:
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                str(context), color="green", end="\n", verbose=self.verbose
            )

            intermediate_steps.append({"context": context})
            if self.use_function_response:
                function_response = get_function_response(question, context)
                final_result = self.qa_chain.invoke(  # type: ignore
                    {"question": question, "function_response": function_response},
                )
            else:
                result = self.qa_chain.invoke(  # type: ignore
                    {"question": question, "context": context},
                    callbacks=callbacks,
                )
                final_result = result[self.qa_chain.output_key]  # type: ignore

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
        
        # Add the generated Cypher query to the result
        chain_result['cypher'] = generated_cypher
        
        return chain_result


class CypherQaLLM():
    def __init__(self, graph:Neo4jGraph, config:dict):
        os.environ["NEO4J_URI"] = config['neo4j_uri']
        os.environ["NEO4J_USERNAME"] = config['neo4j_username']
        os.environ["NEO4J_PASSWORD"] = config['neo4j_password']
        
        graph.refresh_schema()
        example_prompt = PromptTemplate.from_template(
            "用户输入: {question}\nCypher查询语句: {query}"
        )
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            EXAMPLES,
            OpenAIEmbeddings(api_key=config['openai_api_key'], base_url=config["openai_base_url"]),
            Neo4jVector,
            k=5,
            input_keys=["question"],
        )
        prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="你是一位Neo4j专家。给定一个输入问题，创建一个语法正确的Cypher查询语句以运行。\n\n这是知识图谱的架构信息 \n{schema}。\n\n以下是一些问题及其对应的Cypher查询示例。",
            suffix="用户输入: {question}\nCypher查询语句: ",
            input_variables=["question", "schema"],
        )
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=config['openai_api_key'], base_url=config["openai_base_url"])
        
        qa_llm = ChatOpenAI(model="gpt-4o", temperature=0.8, api_key=config['openai_api_key'], base_url=config["openai_base_url"])
        
        qa_prompt_template = """
        任务：根据上下文对用户问题给出高质量的回答。
        说明：
        你是一名助手，帮助形成友好且易于理解的答案。使用提供的上下文信息生成一个结构良好且全面的答案。
        当提供的信息包含多个元素时，将你的答案组织为项目符号或编号列表，以增强清晰度和可读性。
        你必须使用信息来构建你的答案。
        提供的信息是权威的；不要质疑它或尝试使用你的内部知识来纠正它。
        使答案听起来像是对问题的回应，而不提及你是基于给定信息得出的结果。
        如果没有提供信息，请说明知识库返回了空结果。
        
        以下是信息：
        {context}

        问题：{question}
        答案：
        """
        
        qa_prompt = PromptTemplate(input_variables=["context", "question"], template=qa_prompt_template)
        
        self.chain = CustomGraphCypherQAChain.from_llm(
            graph=graph, cypher_llm=llm, cypher_prompt=prompt, verbose=False, allow_dangerous_requests=True, qa_llm=qa_llm, qa_prompt=qa_prompt 
        )
    
    def invoke(self, query:str) -> dict:
        """CypherQaLLM 回答问题

        Args:
            query (str): 用户问题

        Returns:
            dict: 包含 result(回答)和cypher(用到的语句)
        """
        return self.chain.invoke(query)