import yaml
from langchain_community.graphs import Neo4jGraph
from skg_build.utils import (
    rag_extract, batch_construct, 
    get_nodes, get_sub_nodes, save_nodes, save_txt,
    glm_direction, glm_domain, glm_technology, 
    top_domain_construct, top_technology_construct
)
from tqdm import tqdm
from loguru import logger

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    directory = config['papers_path']
    paper_directory = config['papers_path']
    model_directory = "../models"
    zhipuai_api_key = config['zhipuai_api_key']
    device = config['device']
    
    logger.info("连接neo4j数据库...")
    graph = Neo4jGraph(
        url=config['neo4j_uri'],
        username=config['neo4j_username'],
        password=config['neo4j_password']
    )
    logger.info("neo4j数据库连接成功")
    
    graph.query("MATCH (n) detach delete n")
    
    logger.info("构建微观结构...")
    graph_documents = rag_extract(paper_directory, model_directory, zhipuai_api_key, device)
    graph_documents=batch_construct(paper_directory)
    graph.add_graph_documents(graph_documents)
    logger.info("微观结构构建成功")
    
    logger.info("研究方向分类中...")
    problem_nodes = get_nodes("Problem", graph)
    problem_string = save_nodes(problem_nodes, directory)
    
    for retry in range(10):
        try:
            directions = glm_direction(problem_string=problem_string, zhipuai_api_key=zhipuai_api_key)
            domain = glm_domain(directions=directions, zhipuai_api_key=zhipuai_api_key)
            save_txt(directory, "Domain", domain, directions)
            graph_document = top_domain_construct(directory, problem_nodes)
            break
        except Exception as e:
            logger.info(f"Direction Classification encounter {e}, retry {retry+1}")
    
    graph.add_graph_documents([graph_document])
    logger.info("研究方向分类完成")
    
    logger.info("技术路线分类中...")
    direction_nodes = get_nodes("Secondary_Direction", graph)
    graph_documents = []
    for direction_node in tqdm(direction_nodes, total = len(direction_nodes)):
        # 获取接口节点
        method_nodes = get_sub_nodes("Method", direction_node, graph, mode = "direction_method")
        method_string = save_nodes(method_nodes)
        for retry in range(10):
            try:
                # 逆向分类
                technology = glm_technology(method_string=method_string, zhipuai_api_key=zhipuai_api_key)
                save_txt(directory, "Technology", technology)
                # 构建上层结构
                graph_document = top_technology_construct(directory, method_nodes)
                graph_documents.append(graph_document)
                break
            except Exception as e:
                print(f"{direction_node} encounter {e}, retry {retry+1}")
        # 存储
    graph.add_graph_documents(graph_documents)
    logger.info("技术路线分类完成")
    
    logger.info("SurveyKG 构建完成")

if __name__ == "__main__":
    main()
    
