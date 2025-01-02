> 正在思考
# SurveyKG

- [English Version](README.md)
- [中文版](README_zh.md)

## Overview

**SurveyKG** is an advanced knowledge graph system designed to facilitate comprehensive literature reviews within specific academic domains. By automating the extraction and structuring of information from scholarly papers, SurveyKG enables researchers to build high-quality literature databases, visualize complex relationships, and generate insightful reviews. The system integrates powerful tools such as Streamlit for the web interface, Neo4j for graph database management, D3.js for interactive visualizations, and various AI-driven modules for intelligent question-answering.

## Features

- **Automated Literature Processing**: Efficiently extract abstracts, research problems, methods, terminologies, and datasets from PDF papers.
- **Knowledge Graph Construction**: Build a structured and hierarchical knowledge graph using Neo4j, representing relationships between different research components.
- **Interactive Visualization**: Utilize D3.js to create dynamic and interactive visualizations of the knowledge graph.
- **Intelligent Q&A Systems**: Implement multiple QA models (CypherQaLLM, RagQaLLM, LightRagQaLLM, MindMapQaLLM) to provide natural language querying capabilities.
- **Automated Literature Reviews**: Generate high-quality, up-to-date literature reviews based on the constructed knowledge graph.
- **User-Friendly Web Interface**: Leverage Streamlit to offer an intuitive and accessible web interface for interacting with the knowledge graph and QA systems.

## Table of Contents

- [SurveyKG](#surveykg)
  - [Overview](#overview)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Configuration](#configuration)
    - [Example `config.yaml`](#example-configyaml)
    - [Steps](#steps-1)
  - [Deployment](#deployment)
  - [Usage](#usage)
  - [Modules](#modules)
    - [skg\_build](#skg_build)
    - [skg\_qa](#skg_qa)
  - [License](#license)


## Installation

### Prerequisites

- **Python 3.10**
- **Conda** (recommended for environment management)
- **Neo4j Database** (version 5.0 or higher)
- **Git**

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/SurveyKG.git
   cd SurveyKG
   ```

2. **Create and Activate a Conda Environment**

   ```bash
   conda create -n surveykg python=3.10
   conda activate surveykg
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Necessary Models**

   Ensure that all required AI models are downloaded and placed in the `models/` directory as specified in the project structure.
    
    ```bash
    cd models
    git clone https://www.modelscope.cn/maidalun/bce-embedding-base_v1.git
    git clone https://www.modelscope.cn/maidalun/bce-reranker-base_v1.git
    ```


## Configuration

SurveyKG uses a configuration file (`config/config.yaml`) to manage settings such as API keys, database connections, and device configurations.

### Example `config.yaml`

```yaml
openai_api_key: "your_openai_api_key"
openai_base_url: "https://api.openai.com/v1"

zhipuai_api_key: "your_zhipuai_api_key"
device: "cuda"  # or "cpu" or "mps"

neo4j_uri: "bolt://localhost:7687"
neo4j_username: "neo4j"
neo4j_password: "your_neo4j_password"
```

### Steps

1. **Navigate to the `config/` Directory**

   ```bash
   cd config
   ```

2. **Create `config.yaml`**

   Create a file named `config.yaml` and populate it with your configuration details as shown above.

## Deployment

1. **Ensure Neo4j is Running**

   Start your Neo4j database instance and ensure it is accessible using the credentials provided in `config.yaml`.

2. **Build the Knowledge Graph**

   You can build the knowledge graph either through the web interface or by running the build scripts directly.

   - **Using Web Interface**:
     - Run the Streamlit app (see Usage below).
     - Use the sidebar to connect to Neo4j and select the option to rebuild the knowledge graph.

   - **Using Command Line**:
     ```bash
     cd src
     python main.py
     ```

3. **Start the Web Interface**

   ```bash
   cd src
   streamlit run web.py
   ```

   Access the web interface by navigating to `http://localhost:8501` in your browser.

## Usage

1. **Access the Web Interface**

   Open your web browser and navigate to `http://localhost:8501`.

2. **Connect to Neo4j Database**

   - Enter your Neo4j URI, username, and password in the sidebar.
   - Click on "连接Neo4j数据库" to establish the connection.

3. **Build or Use Existing Knowledge Graph**

   - Choose between using an existing graph or rebuilding a new one.
   - If rebuilding, select the appropriate paper directory and initiate the build process.

4. **Explore the Knowledge Graph**

   - Visualize the knowledge graph in the "图谱展示" section.
   - Click on nodes to view detailed information.
   - Use drag and zoom features for better navigation.

5. **Interact with Q&A System**

   - Navigate to the "图谱问答" section.
   - Select the desired QA model.
   - Input your queries and receive structured answers based on the knowledge graph.

## Modules

SurveyKG is modularized into several core components to ensure maintainability and scalability. Each module has its own set of functionalities and responsibilities.

### skg_build

The `skg_build` module is responsible for constructing the knowledge graph from academic papers. It handles tasks such as file processing, information extraction, and graph population.

- **Location**: `src/skg_build/`
- **README**: [skg_build/README.md](src/skg_build/README.md)

### skg_qa

The `skg_qa` module implements various question-answering systems that interact with the knowledge graph. It provides different models tailored for specific types of queries and response formats.

- **Location**: `src/skg_qa/`
- **README**: [skg_qa/README.md](src/skg_qa/README.md)

## License

This project is licensed under the [MIT License](LICENSE).

We hope these steps will help you successfully deploy the SurveyKG project! If you have any questions, please feel free to contact us at [huangkywork@163.com](mailto:huangkywork@163.com)。