import streamlit as st
import pandas as pd
import os
from langchain_community.graphs import Neo4jGraph
from skg_qa.cypher_qa import CypherQaLLM
from skg_qa.naive_rag_qa import RagQaLLM
from skg_qa.light_rag_qa import LightRagQaLLM
from skg_qa.mindmap_qa import MindMapQaLLM
import yaml
from loguru import logger
from tqdm import tqdm

from skg_build.utils import (
    rag_extract, batch_construct, 
    get_nodes, get_sub_nodes, save_nodes, save_txt,
    glm_direction, glm_domain, glm_technology, 
    top_domain_construct, top_technology_construct
)

KG_DISPLAY_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>文献综述知识图谱</title>
    <style>
        /* 页面布局和样式 */

        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
        }}

        /* 设置 SVG 的宽度和高度 */
        svg {{
            width: 100%;
            height: 100%;
            border: 1px solid #ccc;
            background-color: #f9f9f9;
            cursor: grab; /* 指示用户可以拖动图谱 */
        }}

        svg:active {{
            cursor: grabbing;
        }}

        /* 节点的样式 */
        .node circle {{
            stroke: #fff;
            stroke-width: 1.5px;
        }}

        /* 节点标签的样式 */
        .node text {{
            pointer-events: none;
            font: 20px sans-serif;
            text-anchor: middle;
            dominant-baseline: middle;
            fill: #000;
        }}

        /* 关系线的样式 */
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            fill: none;
            marker-end: url(#arrow); /* 添加箭头 */
        }}

        /* 关系标签的样式 */
        .link-label {{
            pointer-events: none;
            font-size: 12px;
            fill: #555;
            text-anchor: middle; /* 居中对齐 */
        }}

        /* 高亮选中节点 */
        .node.selected circle {{
            stroke: #ff0000;
            stroke-width: 1.5px;
        }}

        /* 不同类型节点的颜色 */
        .node.type-Paper circle {{
            fill: #fce38a; /* 蓝色 */
        }}

        .node.type-Problem circle {{
            fill: #02c39a; /* 橙色 */
        }}

        .node.type-Method circle {{
            fill: #028090; /* 绿色 */
        }}

        .node.type-Domain circle {{
            fill: #f18c8e; /* 红色 */
        }}
        
        .node.type-Primary_Direction circle {{
            fill: #f0b7a4; /* 红色 */
        }}
        
        .node.type-Secondary_Direction circle {{
            fill: #f1d1b5; /* 红色 */
        }}
        
        .node.type-Technology circle {{
            fill: #305f72; /* 红色 */
        }}
        
        .node.type-Terminology circle {{
            fill: #eaeaea; /* 红色 */
        }}
        
        .node.type-Dataset circle {{
            fill: #08d9d6; /* 红色 */
        }}

        /* 信息面板样式 */
        .info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            width: 250px;
            padding: 10px;
            border: 1px solid #ccc;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            display: none;
        }}

        .info-panel h2 {{
            margin-top: 0;
            font-size: 20px;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
        }}

        .info-panel p {{
            margin: 10px 0;
        }}

        .close-btn {{
            display: inline-block;
            margin-top: 10px;
            padding: 5px 10px;
            background-color: #f44336;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 14px;
        }}

        .close-btn:hover {{
            background-color: #d32f2f;
        }}
    </style>
</head>
<body>
    <!-- SVG 容器 -->
    <svg></svg>

    <!-- 信息面板 -->
    <div class="info-panel" id="info-panel" style="display: none;">
        <h2 id="info-title">节点信息</h2>
        <p><strong>ID:</strong> <span id="info-id"></span></p>
        <p><strong>标签:</strong> <span id="info-label"></span></p>
        <p><strong>类型:</strong> <span id="info-type"></span></p>
        <p><strong>概括:</strong></p>
        <p id="info-overview"></p>
        <p><strong>描述:</strong></p>
        <p id="info-description"></p>
        <button class="close-btn" id="close-btn">关闭</button>
    </div>

    <!-- 引入 D3.js 库 -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        // 初始化数据：节点和关系
        const fullNodes = {all_nodes};

        const fullLinks = {all_relationships};

        // 当前显示的节点和链接（用于过滤）
        let currentNodes = fullNodes;
        let currentLinks = fullLinks;

        // 选择 SVG 并获取其宽高
        const svg = d3.select("svg"),
              width = window.innerWidth,
              height = window.innerHeight;

        svg
            .attr("width", width)
            .attr("height", height);

        // 添加定义部分，用于箭头标记
        svg.append('defs').append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25) // 调整箭头位置
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#999');

        // 添加背景 rect 用于监听双击重置
        svg.append("rect")
            .attr("width", width)
            .attr("height", height)
            .attr("fill", "none")
            .attr("pointer-events", "all")
            .lower() // 将其置于图层底部
            .on("dblclick", resetGraph);

        // 创建一个组元素，用于所有可缩放内容
        const graphGroup = svg.append("g");

        // 创建一个力导向图仿真器
        const simulation = d3.forceSimulation(currentNodes)
            .force("link", d3.forceLink(currentLinks)
                .id(d => d.id)
                .distance(150)) // 链接的距离
            .force("charge", d3.forceManyBody().strength(-300)) // 节点之间的斥力
            .force("center", d3.forceCenter(width / 2, height / 2)) // 将图居中
            .force("collision", d3.forceCollide().radius(50)); // 防止节点重叠

        // 添加关系线
        const link = graphGroup.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(currentLinks)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", 2);

        // 添加关系标签
        const linkLabels = graphGroup.append("g")
            .attr("class", "link-labels")
            .selectAll("text")
            .data(currentLinks)
            .enter().append("text")
            .attr("class", "link-label")
            .attr("font-size", 12)
            .attr("fill", "#555")
            .text(d => d.label);

        // 添加节点
        const node = graphGroup.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(currentNodes)
            .enter().append("g")
            .attr("class", d => `node type-${{d.type}}`) // 根据类型添加类名
            .call(drag(simulation));

        // 添加节点圆形
        node.append("circle")
            .attr("r", 50)
            .attr("fill", d => getNodeColor(d.type)) // 根据类型设置颜色
            .on("click", function(event, d) {{
                // 1. 取消所有节点的选中状态
                d3.selectAll('.node').classed('selected', false);
                // 2. 选中当前点击的节点
                d3.select(this.parentNode).classed("selected", true);
                // 3. 显示信息面板
                showInfoPanel(d);
            }})
            .on("dblclick", function(event, d) {{
                // 阻止双击事件传播，以避免触发点击事件
                event.stopPropagation();
                // 切换过滤状态
                toggleFilter(d);
            }});

        // 添加节点标签
        node.append("text")
            .text(d => d.label);

        // 更新节点和关系的位置
        simulation.on("tick", () => {{
            // 更新关系线的位置
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            // 更新关系标签的位置和旋转
            linkLabels
                .attr("transform", function(d) {{
                    const midX = (d.source.x + d.target.x) / 2;
                    const midY = (d.source.y + d.target.y) / 2;
                    const angle = Math.atan2(d.target.y - d.source.y, d.target.x - d.source.x) * 180 / Math.PI;
                    return `translate(${{midX}},${{midY}}) rotate(${{angle}})`;
                }});

            // 更新节点的位置
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        // 拖拽功能
        function drag(simulation) {{
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}

            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}

            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }}

        // 信息面板元素选择
        const infoPanel = document.getElementById('info-panel');
        const infoTitle = document.getElementById('info-title');
        const infoId = document.getElementById('info-id');
        const infoLabel = document.getElementById('info-label');
        const infoType = document.getElementById('info-type');
        const infoOverview = document.getElementById('info-overview');
        const infoDescription = document.getElementById('info-description');
        const closeBtn = document.getElementById('close-btn');

        // 显示信息面板函数
        function showInfoPanel(nodeData) {{
            infoTitle.textContent = `节点信息: ${{nodeData.label}}`;
            infoId.textContent = nodeData.id;
            infoLabel.textContent = nodeData.label;
            infoType.textContent = nodeData.type;
            infoOverview.textContent = nodeData.overview;
            infoDescription.textContent = nodeData.description;
            infoPanel.style.display = 'block';
        }}

        // 关闭信息面板
        closeBtn.addEventListener('click', () => {{
            infoPanel.style.display = 'none';
            // 取消所有节点的选中状态
            d3.selectAll('.node').classed("selected", false);
        }});

        // 窗口大小调整
        window.addEventListener("resize", () => {{
            const newWidth = document.querySelector('.graph-container').clientWidth;
            const newHeight = window.innerHeight - document.querySelector('header').clientHeight - 80;
            svg.attr("width", newWidth).attr("height", newHeight);
            // 更新背景 rect 的大小
            svg.select("rect")
                .attr("width", newWidth)
                .attr("height", newHeight);
            // 更新仿真器中心力
            simulation.force("center", d3.forceCenter(newWidth / 2, newHeight / 2));
            simulation.alpha(0.3).restart();
        }});

        // 变量用于跟踪过滤状态
        let isFiltered = false;
        let filteredNodeId = null;

        // 实现双击过滤功能
        function toggleFilter(nodeData) {{
            if (isFiltered && filteredNodeId === nodeData.id) {{
                // 重置图谱显示所有节点和链接
                d3.selectAll('.node').style('display', 'block');
                d3.selectAll('.link').style('display', 'block');
                d3.selectAll('.link-label').style('display', 'block');
                isFiltered = false;
                filteredNodeId = null;
            }} else {{
                // 只显示点击节点和与之直接相连的节点及其链接
                const connectedNodes = new Set();
                connectedNodes.add(nodeData.id);
                currentLinks.forEach(l => {{
                    if (l.source.id === nodeData.id) {{
                        connectedNodes.add(l.target.id);
                    }} else if (l.target.id === nodeData.id) {{
                        connectedNodes.add(l.source.id);
                    }}
                }});

                // 显示相关节点，隐藏其他节点
                d3.selectAll('.node').style('display', d => connectedNodes.has(d.id) ? 'block' : 'none');
                // 显示相关链接，隐藏其他链接
                d3.selectAll('.link').style('display', d => d.source.id === nodeData.id || d.target.id === nodeData.id ? 'block' : 'none');
                // 显示相关关系标签，隐藏其他标签
                d3.selectAll('.link-label').style('display', d => d.source.id === nodeData.id || d.target.id === nodeData.id ? 'block' : 'none');

                isFiltered = true;
                filteredNodeId = nodeData.id;
            }}
        }}

        // 重置图谱显示
        function resetGraph() {{
            d3.selectAll('.node').style('display', 'block');
            d3.selectAll('.link').style('display', 'block');
            d3.selectAll('.link-label').style('display', 'block');
            isFiltered = false;
            filteredNodeId = null;
        }}

        // 获取节点颜色函数
        function getNodeColor(type) {{
            const colorMap = {{
                'A': '#1f77b4', // 蓝色
                'B': '#ff7f0e', // 橙色
                'C': '#2ca02c', // 绿色
                'D': '#d62728'  // 红色
            }};
            return colorMap[type] || '#69b3a2'; // 默认颜色
        }}

        // 添加缩放行为
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10]) // 缩放比例范围
            .on("zoom", (event) => {{
                graphGroup.attr("transform", event.transform);
            }});

        // 应用缩放行为到 SVG
        svg.call(zoom);
    </script>
</body>
</html>

"""

def generate_html(graph):
    all_nodes_and_relationships = graph.query(
        """
        MATCH (n)-[r]->(m)
        WITH n, r, m
        LIMIT 1000
        RETURN 
            labels(n) AS 起始节点标签, 
            properties(n) AS 起始节点属性,
            type(r) AS 关系类型, 
            labels(m) AS 终止节点标签, 
            properties(m) AS 终止节点属性
        """
    )
    
    def extract_data(item):
        source_node_type = item['起始节点标签'][0]
        source_node_id = item['起始节点属性'].get("id") if item['起始节点属性'].get("id") else ""
        source_node_desc = item['起始节点属性'].get("description") if item['起始节点属性'].get("description") else ""
        
        target_node_type = item['终止节点标签'][0]
        target_node_id = item['终止节点属性'].get("id") if item['终止节点属性'].get("id") else ""
        target_node_desc = item['终止节点属性'].get("description") if item['终止节点属性'].get("description") else ""
        
        relationship_type = item['关系类型']
        
        return source_node_type, source_node_id, source_node_desc, target_node_type, target_node_id, target_node_desc, relationship_type
    
    all_nodes = []
    all_relationships = []

    nodes_record = {}
    relationships_record = {}

    idx = 0
    for item in all_nodes_and_relationships:
        source_node_type, source_node_id, source_node_desc, target_node_type, target_node_id, target_node_desc, relationship_type = extract_data(item)
        
        if (source_node_id, source_node_type) not in nodes_record:
            nodes_record[(source_node_id, source_node_type)] = str(idx)
            all_nodes.append(
                {"id": str(idx), "overview": source_node_id, "label": source_node_id[:5], "type": source_node_type, "description": source_node_desc}
            )
            idx += 1
        
        if (target_node_id, target_node_type) not in nodes_record:
            nodes_record[(target_node_id, target_node_type)] = str(idx)
            all_nodes.append(
                {"id": str(idx), "overview": target_node_id, "label": target_node_id[:5], "type": target_node_type, "description": target_node_desc}
            )
            idx += 1
        
        if ((source_node_id, source_node_type), (target_node_id, target_node_type)) not in relationships_record:
            relationships_record[((source_node_id, source_node_type), (target_node_id, target_node_type))] = 1
            all_relationships.append(
                {"source": nodes_record[(source_node_id, source_node_type)], "target": nodes_record[(target_node_id, target_node_type)], "label": relationship_type}
            )
    
    return KG_DISPLAY_HTML.format(all_nodes=all_nodes, all_relationships=all_relationships)

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def build_new_kg(config, papers_path):
    directory = papers_path
    paper_directory = papers_path
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

config = load_config()

st.markdown("# Survey Knowledge Graph")

with st.sidebar:
    st.markdown("# 问答模型")
    openai_api_key = st.text_input(label="请输入OpenAI密钥:", type="password", value=config['openai_api_key'])
    openai_base_url = st.text_input(label="请输入Base Url:", value=config['openai_base_url'])
    
    st.markdown("# Neo4j")
    neo4j_uri = st.text_input(label="请输入Neo4j Uri:", value=config['neo4j_uri'])
    neo4j_username = st.text_input(label="请输入Neo4j Username:", value=config['neo4j_username'])
    neo4j_pwd = st.text_input(label="请输入Neo4j Password:", type="password", value=config['neo4j_password'])
    
    if neo4j_uri and neo4j_username and neo4j_pwd:
        def connect_neo4j():
            try:
                st.session_state["graph"] = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_pwd)
                st.success("Neo4j数据库已连接成功。")
            except:
                st.warning(f"连接失败，请确保Neo4j数据库已启动且输入正确。", icon="💡")
            return
        st.button("连接Neo4j数据库", on_click=connect_neo4j)
    
    if "graph" in st.session_state:
        st.markdown("# 知识图谱")
        def file_selector(folder_path):
            foldernames = ["None"] + os.listdir(folder_path)
            selected_foldername = st.selectbox('选择生成综述的领域', foldernames)
            return os.path.join(folder_path, selected_foldername)
        
        choose_kg = st.radio(label="选择知识图谱", options=["使用现有图谱", "重新创建图谱"], index=0, horizontal=True)
        
        if choose_kg == "重新创建图谱": 
            foldername = file_selector(folder_path="../papers")
            if foldername != os.path.join("../papers", "None"):
                df = pd.DataFrame(os.listdir(foldername), columns=["文章列表"])
                st.dataframe(df)
            
            def build_new():
                try:
                    with st.spinner("正在构建图谱，请稍后..."):
                        build_new_kg(config=config, papers_path=foldername)
                        st.success("图谱创建成功。")
                except Exception as e:
                    st.warning(f"图谱创建失败。{e}", icon="⚠️")
            
            if foldername != os.path.join("../papers", "None"):
                st.button(label="确认构建", on_click=build_new)
            
        
st.markdown("## 图谱展示")

if "graph" in st.session_state:
    if "whole_graph_display" not in st.session_state:
        st.session_state["whole_graph_display"] = generate_html(graph=st.session_state["graph"])
    st.components.v1.html(st.session_state["whole_graph_display"], height=800)

st.divider()

st.markdown("## 图谱问答")

if "graph" in st.session_state:

    qa_model = st.selectbox("选择回答模型", ["None", "CypherQaLLM", "MindMapQaLLM"], index=0)

    if qa_model == "CypherQaLLM":
        if "CypherQaLLM" not in st.session_state:
            with st.spinner("正在创建CypherQaLLM，请稍后..."):
                st.session_state["CypherQaLLM"] = CypherQaLLM(graph=st.session_state["graph"], config=config)
        QaLLM = st.session_state["CypherQaLLM"]
    elif qa_model == "RagQaLLM":
        if "RagQaLLM" not in st.session_state:
            with st.spinner("正在创建RagQaLLM，请稍后..."):
                st.session_state["RagQaLLM"] = RagQaLLM(graph=st.session_state["graph"], config=config, exclude=["Chunk"])
        QaLLM = st.session_state["RagQaLLM"]
    elif qa_model == "LightRagQaLLM":
        if "LightRagQaLLM" not in st.session_state:
            with st.spinner("正在创建LightRagQaLLM，请稍后..."):
                st.session_state["LightRagQaLLM"] = LightRagQaLLM(graph=st.session_state["graph"], config=config, exclude=["Chunk"])
        QaLLM = st.session_state["LightRagQaLLM"]
    elif qa_model == "MindMapQaLLM":
        if "MindMapQaLLM" not in st.session_state:
            with st.spinner("正在创建MindMapQaLLM，请稍后..."):
                st.session_state["MindMapQaLLM"] = MindMapQaLLM(graph=st.session_state["graph"], config=config, exclude=["Chunk"])
        QaLLM = st.session_state["MindMapQaLLM"]

    if "messages" not in st.session_state and qa_model != "None":
        st.session_state["messages"] = [{"role": "assistant",
                                        "content": "您好，我是SurveyKG助手，有什么可以帮助您的吗？"}]
    
    def _clear_history():
        st.session_state["messages"] = [{"role": "assistant",
                                        "content": "您好，我是SurveyKG助手，有什么可以帮助您的吗？"}]
    
    if qa_model in st.session_state:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        user_input = st.chat_input(placeholder="输入问题")
    
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
                st.session_state["messages"].append({"role": "user", "content": user_input})
            
            with st.spinner("AI正在思考，请稍后......"):
                answer = "我不知道答案。"
                cypher = ""
                documents = ""
                context = ""
                for _ in range(5):
                    try:
                        response = QaLLM.invoke(user_input)
                        answer = response.get("result", "")
                        
                        cypher = response.get("cypher", "")
                        documents = response.get("documents", "")
                        context = response.get("context", "")
                        subgraph = response.get("subgraph", "")
                        break
                    except Exception as e:
                        print(e)
                        continue
            
            with st.chat_message("assistant"):
                st.write(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
            
            if cypher:
                with st.popover("CypherQaLLM使用的cypher查询语句"):
                    st.code(cypher, language="cypher")
            elif documents:
                with st.popover("RagQaLLM检索到的相关文档"):
                    st.text(documents)
            elif context and not subgraph:
                with st.popover("LightRagQaLLM检索到的相关上下文"):
                    st.markdown("#### 节点信息数据表")
                    st.dataframe(context[0])
                    st.markdown("#### 关系信息数据表")
                    st.dataframe(context[1])
            elif context and subgraph:
                with st.popover("MindMapQaLLM检索到的相关上下文"):
                    st.markdown("#### 节点信息数据表")
                    st.dataframe(context[0])
                    st.markdown("#### 关系信息数据表")
                    st.dataframe(context[1])
                st.components.v1.html(KG_DISPLAY_HTML.format(all_nodes=subgraph[0], all_relationships=subgraph[1]), height=800)


    if "messages" in st.session_state and len(st.session_state["messages"]) > 1:
        st.button(label="清空对话", on_click=_clear_history)