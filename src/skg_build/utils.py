import os
import json
import fnmatch
import PyPDF2
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from openai import OpenAI
from zhipuai import ZhipuAI
from pathlib import Path
from tqdm import tqdm

from skg_build.prompts import *

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader # 读取text文件
from langchain_text_splitters import RecursiveCharacterTextSplitter # 将读取的文件拆分为chunk
from langchain_community.embeddings import HuggingFaceEmbeddings # 读取huggingface的embedding模型
from langchain_community.vectorstores import FAISS # 把embedding模型的编码结果储存为向量数据库
from langchain_community.cross_encoders import HuggingFaceCrossEncoder # 读取huggingface的cross_embedding模型
from langchain.retrievers.document_compressors import CrossEncoderReranker # 设置reranker模型的重排方法
from langchain.retrievers import ContextualCompressionRetriever # 整合embedding和reranker

# 构造 chatgpt + rag
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate


def find_files(directory, pattern):
    # pattern 为文件后缀
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, "*."+pattern):
            file_paths.append(os.path.join(root, filename))
    return file_paths

def extract_abstract(pdf_path):
    # 打开 PDF 文件
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        abstract = None
        
        # 遍历每一页，提取文本
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            if text:
                # 使用正则表达式查找 "Abstract" 部分
                # 改进：确保匹配 "Abstract" 到第一个空行（或遇到其他标题词）
                match = re.search(r"abstract[\s:]*\s*(.*?)(?=\n\s*\n|(1 )?Introduction|\Z)", text, re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1).strip()
                    break
        
        if abstract:
            return abstract
        else:
            return "Abstract not found"

def glm_emphasis(abstract, zhipuai_api_key):
    client = ZhipuAI(api_key=zhipuai_api_key)  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-flash",  # 请填写您要调用的模型名称
        temperature=0.3,
        messages=[
            {"role": "system", "content": EMPHASIS_SYSTEM},
            {"role": "user", "content": EMPHASIS_USER.format(text=abstract)},
        ],
    )
    return response.choices[0].message.content

def spliter(file_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
    )

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    texts = text_splitter.split_documents(docs)
    # print(f"该论文共计{sum([len(doc.page_content) for doc in docs])}词。")
    return texts

def extract_micro_structure(paper_path, extracts, targets, emphasis, chain):
    for target, prompt in targets.items():
        target_source = extracts[f"{target}_source"]
        order = EMPHASIS_PROMPT.format(emphasis=emphasis) + SOURCE_PROMPT.format(source=target_source) + prompt

        for retry in range(5):
            try:
                extracts[target] = chain.invoke(order)
                break
            except Exception as e:
                logger.info(f"{paper_path} -- {target} -- {e}")
                extracts[target] = f"{paper_path} -- {target} -- {e}"

    return paper_path, extracts

def rag_extract(paper_directory, model_directory, zhipuai_api_key, device):
    # 抽取任务
    targets = {"problem": PROBLEM_PROMPT, "method": METHOD_PROMPT, "terminology": TERMINOLOGY_PROMPT, "dataset": DATASET_PROMPT}
    # 加载模型
    embeddings_model = HuggingFaceEmbeddings(model_name=os.path.join(model_directory, "bce-embedding-base_v1"),
                                        model_kwargs={"device": device},
                                        encode_kwargs={"normalize_embeddings": True})
    reranker = HuggingFaceCrossEncoder(model_name=os.path.join(model_directory, "bce-reranker-base_v1"),
                                    model_kwargs={"device": device})
    compressor = CrossEncoderReranker(model=reranker, top_n=3)
    llm = ChatZhipuAI(model="glm-4-flash",
                temperature=0.5,
                api_key=zhipuai_api_key,
                )
    chain = llm | JsonOutputParser()
    
    # 批量抽取
    paper_paths = find_files(paper_directory, "pdf")
    
    texts = []
    for paper_path in tqdm(paper_paths, desc="切分chunk"):
        texts += spliter(paper_path)
    
    logger.info("正在构建向量数据库...")
    db = FAISS.from_documents(texts, embeddings_model)
    logger.info("向量数据库构建完成")
    
    extracts_dict = {}
    
    for paper_path in tqdm(paper_paths, desc="检索相关文档"):
        try:
            extracts = {}
            abstract = extract_abstract(paper_path)
            extracts["abstract"] = abstract
            emphasis = glm_emphasis(abstract, zhipuai_api_key)
            
            retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 10, "filter": {"source": paper_path}})
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
            
            for target, prompt in targets.items():
                order = EMPHASIS_PROMPT.format(emphasis=emphasis) + prompt
                result = compression_retriever.invoke(order)
                result = [r.page_content for r in result]
                
                extracts[f"{target}_source"] = result

            extracts_dict[paper_path] = extracts
        except Exception as e:
            logger.info(f"{paper_path} encounter {e}")
    
    # 使用多线程抽取微观结构
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {
            executor.submit(extract_micro_structure, paper_path, extracts, targets, emphasis, chain): paper_path
            for paper_path, extracts in extracts_dict.items()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            paper_path, extracts = future.result()
            extracts_dict[paper_path] = extracts

    # 保存到json文件
    for paper_path, extracts in tqdm(extracts_dict.items(), desc="保存到json文件"):
        paper_path_json = paper_path.replace(".pdf", ".json")
        with open(paper_path_json, "w", encoding="utf-8") as f:
            json.dump(extracts, f, ensure_ascii=False, indent=4)
        
            

def base_construct(json_path):
    def add_node(target, extract, sources):
        node = Node(id=extract["name"], type=target.title(), properties={"description": extract["description"]})
        nodes[target].append(node)
        relationships[f"paper_{target}"].append(Relationship(source=paper_node, target=node, type=relation_types[f"paper_{target}"]))
        # print(f"添加{target}节点")
        # print(f"添加关系paper_{target}")
        for source in sources:
            source_node = Node(id=source, type="Source")
            nodes["source"].append(source_node)
            relationships[f"{target}_source"].append(Relationship(source=node, target=source_node, type="reference"))
            # print("添加source节点")
            # print(f"添加关系{target}_source")
    
    extracts = json.load(open(json_path, "r", encoding="utf-8"))
    
    paper_name = os.path.basename(json_path)
    paper_name = paper_name.replace(".json", "")
    paper_node = Node(id=paper_name, type="Paper", properties={"description": extracts["abstract"]})
    
    node_types = ["problem", "method", "terminology", "dataset"]
    nodes = {"paper": [paper_node], "problem": [], "method": [], "terminology": [], "dataset": [], "source": []}
    relation_types = {"paper_problem": "research", "paper_method": "use", "paper_terminology": "mention", "paper_dataset": "base on"}
    relationships = {"paper_problem": [], "paper_method": [], "paper_terminology": [], "paper_dataset": [], "problem_source": [], "method_source": [], "terminology_source": [], "dataset_source": [], "method_problem": [], "dataset_problem": []}
    # print("添加paper节点")
    
    for target in node_types:
        sources = extracts[target+"_source"]
        if isinstance(extracts[target], list):
            for extract in extracts[target]:
                add_node(target, extract, sources)
        else:
            extract = extracts[target]
            add_node(target, extract, sources)
    for problem_node in nodes["problem"]:
        for method_node in nodes["method"]:
            relationships["method_problem"].append(Relationship(source=method_node, target=problem_node, type="solve"))
            # print("添加关系method_problem")
        for dataset_node in nodes["dataset"]:
            relationships["dataset_problem"].append(Relationship(source=dataset_node, target=problem_node, type="use for"))
            # print("添加关系dataset_problem")
    
    all_nodes = []
    all_relationships = []
    for node in nodes.values():
        all_nodes.extend(node)
    for relationship in relationships.values():
        all_relationships.extend(relationship)
    return GraphDocument(nodes = all_nodes,
                        relationships = all_relationships,
                        source = Document(page_content = extracts["abstract"]))

def batch_construct(json_directory):
    graph_documents = []
    paper_paths = find_files(json_directory, "json")
    for paper_path in tqdm(paper_paths, total=len(paper_paths)):
        try:
            print(f"正在为{paper_path}构建论文结构")
            graph_document = base_construct(paper_path)
            if graph_document:
                graph_documents.append(graph_document)
            print(f"{paper_path}构建成功")
        except:
            print(f"{paper_path}构建失败")
    return graph_documents

def glm_direction(problem_string, zhipuai_api_key):
    client = ZhipuAI(api_key=zhipuai_api_key)
    
    completion = client.chat.completions.create(
        model="glm-4-flash",
        temperature=1,
        messages=[
            {"role": "system", "content": DIRECTION_SYSTEM},
            {"role": "user", "content": DIRECTION_USER.format(problem_string=problem_string)},
        ]
    )
    return completion.choices[0].message.content

def glm_domain(directions, zhipuai_api_key):
    client = ZhipuAI(api_key=zhipuai_api_key)
    
    completion = client.chat.completions.create(
        model="glm-4-flash",
        temperature=1,
        messages=[
            {"role": "system", "content": DOMAIN_SYSTEM},
            {"role": "user", "content": DOMAIN_USER.format(directions=directions)},
        ]
    )
    return completion.choices[0].message.content

def save_txt(directory, file_name, *texts):
    save_path = os.path.join(directory, f"{file_name}.txt")
    with open(save_path, "w", encoding="utf-8") as file:
        for text in texts:
            file.write(text + "\n\n")

def get_nodes(node_type, graph):
    nodes = graph.query(
                        f"""
                        MATCH (n:{node_type}) return n
                        """
                    )
    nodes = [Node(id=node['n']['id'], type=node_type, properties={"description": node['n']['description']}) for node in nodes]
    return nodes

def get_sub_nodes(node_type, node, graph, mode="sub"):
    if mode == "sub":
        sub_nodes = graph.query(
            f"""
            MATCH (a:{node.type})-->(n:{node_type})
            WHERE a.id='{node.id}'
            RETURN n
            """
        )
    elif mode == "direction_method":
        sub_nodes = graph.query(
            f"""
            MATCH (a:{node.type})-->(b:Problem)<--(n:{node_type})
            WHERE a.id='{node.id}'
            RETURN n
            """
        )
    sub_nodes = [Node(id=sub_node['n']['id'], type=node_type, properties={"description": sub_node['n']['description']}) for sub_node in sub_nodes]
    return sub_nodes

def save_nodes(nodes, directory = None):
    node_string = "\n".join([f"{idx+1}. {node.id} {node.properties['description']}" for idx, node in enumerate(nodes)])
    if directory:
        save_path = os.path.join(directory, f"{nodes[0].type}.txt")
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(node_string)
    return node_string

def top_domain_construct(directory, problem_nodes):
    primary_direction_node = []
    secondary_direction_node = []
    domain_direction_relationship = []
    f_s_relationship = []
    direction_problem_relationship= []
    
    with open(os.path.join(directory, 'Domain.txt'), encoding="utf-8") as file:
        info = file.read()
        info += "\n"
    for line in info.splitlines():
        # 跳过空行
        if not line:
            continue
        
        # 读取到Domain
        domain_match = re.match(r"Domain: (.+)", line)
        if domain_match:
            domain = domain_match.group(1).strip()
            continue
        # 读取到Domain的描述，创建Domain节点
        domain_description_match = re.match(r"Detailed description of domain: (.+)", line)
        if domain_description_match:
            domain_node = Node(id=domain, type="Domain", properties={"description": domain_description_match.group(1).strip()})
            continue
        
        # 读取到Primary Direction
        primary_direction_match = re.match(r"Primary Direction \d: (.+)", line)
        if primary_direction_match:
            primary_direction = primary_direction_match.group(1).strip()
            continue
        # 读取到Primary Direction的描述
        primary_direction_description_match = re.match(r"Detailed description of primary direction \d: (.+)", line)
        if primary_direction_description_match:
            # 创建Primary Direction节点
            current_direction = Node(id=primary_direction, type="Primary_Direction", properties={"description": primary_direction_description_match.group(1).strip()})
            primary_direction_node.append(current_direction)
            # 创建Primary Direction与Domain的关系
            domain_direction_relationship.append(Relationship(source=domain_node, target=current_direction, type="include"))
            continue
        
        # 读取到Secondary Direction
        if re.match(r"Secondary Direction \d\.\d: (.+)", line):
            secondary_direction = line.split(": ")[1].strip()
            continue
        # 读取到Secondary Direction的描述
        secondary_direction_description_match = re.match(r"Detailed description of secondary direction \d\.\d: (.+)", line)
        if secondary_direction_description_match:
            # 创建Secondary Direction节点
            current_direction = Node(id=secondary_direction, type="Secondary_Direction", properties={"description": secondary_direction_description_match.group(1).strip()})
            secondary_direction_node.append(current_direction)
            # 创建Secondary Direction与Primary Direction的关系
            f_s_relationship.append(Relationship(source=primary_direction_node[-1], target=current_direction, type="include"))
            continue
        
        # 读取到Problem
        number_match = re.match(r"Number: (.+)", line)
        if number_match:
            number = json.loads(number_match.group(1).strip())
            # 链接Problem到当前的Secondary Direction
            current_p_d_relationship = [Relationship(source=secondary_direction_node[-1], target=problem_nodes[idx-1], type="include") for idx in number]
            direction_problem_relationship.extend(current_p_d_relationship)
    
    return GraphDocument(nodes=[domain_node] + primary_direction_node + secondary_direction_node + problem_nodes,
                        relationships=domain_direction_relationship + f_s_relationship + direction_problem_relationship,
                        source=Document(page_content=""))

def glm_technology(method_string, zhipuai_api_key):
    client = ZhipuAI(api_key=zhipuai_api_key)
    
    completion = client.chat.completions.create(
        model="glm-4-flash",
        temperature=0.5,
        messages=[
            {"role": "system", "content": TECHNOLOGY_SYSTEM},
            {"role": "user", "content": TECHNOLOGY_USER.format(method_string=method_string)},
        ]
    )
    return completion.choices[0].message.content

def top_technology_construct(directory, method_nodes):
    technology_nodes = []
    technology_method_relationships = []
    
    with open(os.path.join(directory, 'Technology.txt'), encoding="utf-8") as file:
        info = file.read()
        info += "\n"
    for line in info.splitlines():
        # 跳过空行
        if not line:
            continue
        
        # 读取到Technical Route
        technology_match = re.match(r"Technical Route \d: (.+)", line)
        if technology_match:
            technology = technology_match.group(1).strip()
            continue
        # 读取到Technical Route的描述，创建technology节点
        technology_description_match = re.match(r"Detailed description of technical route \d: (.+)", line)
        if technology_description_match:
            current_technology = Node(id=technology, type="Technology", properties={"description": technology_description_match.group(1).strip()})
            technology_nodes.append(current_technology)
            continue
        
        # 读取到Method
        number_match = re.match(r"Number: (.+)", line)
        if number_match:
            number = json.loads(number_match.group(1).strip())
            # 链接Method到当前的Technology
            current_t_m_relationship = [Relationship(source=technology_nodes[-1], target=method_nodes[idx-1], type="include") for idx in number]
            technology_method_relationships.extend(current_t_m_relationship)
    
    return GraphDocument(nodes=method_nodes+technology_nodes,
                        relationships=technology_method_relationships,
                        source=Document(page_content=""))