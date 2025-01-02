# SurveyKG

- [English Version](README.md)
- [中文版](README_zh.md)

## 概述

**SurveyKG** 是一个先进的知识图谱系统，旨在促进特定学术领域内的全面文献综述。通过自动化从学术论文中提取和结构化信息，SurveyKG 使研究人员能够构建高质量的文献数据库、可视化复杂关系并生成有洞察力的综述。该系统集成了强大的工具，如用于网页界面的 Streamlit、用于图数据库管理的 Neo4j、用于交互式可视化的 D3.js 以及各种基于AI的智能问答模块。

## 功能

- **自动化文献处理**：高效地从PDF论文中提取摘要、研究问题、方法、术语和数据集。
- **知识图谱构建**：使用 Neo4j 构建结构化和分层的知识图谱，表示不同研究组件之间的关系。
- **交互式可视化**：利用 D3.js 创建动态和交互式的知识图谱可视化。
- **智能问答系统**：实现多种QA模型（CypherQaLLM、RagQaLLM、LightRagQaLLM、MindMapQaLLM），提供自然语言查询功能。
- **自动化文献综述**：基于构建的知识图谱生成高质量、最新的文献综述。
- **用户友好的网页界面**：利用 Streamlit 提供直观且易于访问的网页界面，用于与知识图谱和QA系统进行交互。

## 目录

- [SurveyKG](#surveykg)
  - [概述](#概述)
  - [功能](#功能)
  - [目录](#目录)
  - [安装](#安装)
    - [前提条件](#前提条件)
    - [步骤](#步骤)
  - [配置](#配置)
    - [示例 `config.yaml`](#示例-configyaml)
    - [步骤](#步骤-1)
  - [部署](#部署)
  - [使用方法](#使用方法)
  - [模块](#模块)
    - [skg\_build](#skg_build)
    - [skg\_qa](#skg_qa)
  - [许可证](#许可证)

## 安装

### 前提条件

- **Python 3.10**
- **Conda**（推荐用于环境管理）
- **Neo4j 数据库**（版本 5.0 或更高）
- **Git**

### 步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/yourusername/SurveyKG.git
   cd SurveyKG
   ```

2. **创建并激活 Conda 环境**

   ```bash
   conda create -n surveykg python=3.10
   conda activate surveykg
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

4. **下载必要的模型**

   确保所有需要的AI模型已下载并放置在项目结构中指定的 `models/` 目录下。

    ```bash
    cd models
    git clone https://www.modelscope.cn/maidalun/bce-embedding-base_v1.git
    git clone https://www.modelscope.cn/maidalun/bce-reranker-base_v1.git
    ```

## 配置

SurveyKG 使用配置文件 (`config/config.yaml`) 来管理设置，如API密钥、数据库连接和设备配置。

### 示例 `config.yaml`

```yaml
openai_api_key: "your_openai_api_key"
openai_base_url: "https://api.openai.com/v1"

zhipuai_api_key: "your_zhipuai_api_key"
device: "cuda"  # 或 "cpu" 或 "mps"

neo4j_uri: "bolt://localhost:7687"
neo4j_username: "neo4j"
neo4j_password: "your_neo4j_password"
```

### 步骤

1. **导航到 `config/` 目录**

   ```bash
   cd config
   ```

2. **创建 `config.yaml`**

   创建一个名为 `config.yaml` 的文件，并按照上面的示例填充您的配置信息。

## 部署

1. **确保 Neo4j 正在运行**

   启动您的 Neo4j 数据库实例，并确保可以使用 `config.yaml` 中提供的凭据进行访问。

2. **构建知识图谱**

   您可以通过网页界面或直接运行构建脚本来构建知识图谱。

   - **使用网页界面**：
     - 运行 Streamlit 应用（见下文使用方法）。
     - 使用侧边栏连接到 Neo4j，并选择重建知识图谱的选项。

   - **使用命令行**：
     ```bash
     cd src
     python main.py
     ```

3. **启动网页界面**

   ```bash
   cd src
   streamlit run web.py
   ```

   在浏览器中访问 `http://localhost:8501` 以打开网页界面。

## 使用方法

1. **访问网页界面**

   打开您的网页浏览器，导航到 `http://localhost:8501`。

2. **连接到 Neo4j 数据库**

   - 在侧边栏中输入您的 Neo4j URI、用户名和密码。
   - 点击“连接Neo4j数据库”以建立连接。

3. **构建或使用现有的知识图谱**

   - 选择使用现有图谱或重建新的图谱。
   - 如果选择重建，选择合适的论文目录并启动构建过程。

4. **探索知识图谱**

   - 在“图谱展示”部分可视化知识图谱。
   - 点击节点查看详细信息。
   - 使用拖拽和缩放功能进行更好的导航。

5. **与问答系统交互**

   - 导航到“图谱问答”部分。
   - 选择所需的QA模型。
   - 输入您的查询，并根据知识图谱接收结构化的答案。

## 模块

SurveyKG 模块化为多个核心组件，以确保可维护性和可扩展性。每个模块都有其独特的功能和职责。

### skg_build

`skg_build` 模块负责从学术论文构建知识图谱。它处理文件处理、信息提取和图谱填充等任务。

- **位置**：`src/skg_build/`
- **README**： [skg_build/README.md](src/skg_build/README.md)

### skg_qa

`skg_qa` 模块实现了与知识图谱交互的各种问答系统。它提供了针对特定类型查询和响应格式的不同模型。

- **位置**：`src/skg_qa/`
- **README**： [skg_qa/README.md](src/skg_qa/README.md)

## 许可证

本项目采用 [MIT 许可证](LICENSE) 进行许可。

我们希望这些步骤能帮助您成功部署 SurveyKG 项目！如果您有任何问题，请随时通过 [huangkywork@163.com](mailto:huangkywork@163.com) 联系我们。
