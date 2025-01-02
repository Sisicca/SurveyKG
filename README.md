# SurveyKG

### 项目部署教程

#### 环境准备

- Python 3.10 及以上版本
- Neo4j 数据库
- OpenAI API 密钥
- ZhipuAI API 密钥

#### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-repo/SurveyKG.git
   cd SurveyKG
   ```

2. **环境初始化**
    ```bash
    conda create -n new_env python=3.10
    conda activate new_env
    ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **下载模型**
    ```bash
    cd models
    git clone https://www.modelscope.cn/maidalun/bce-embedding-base_v1.git
    git clone https://www.modelscope.cn/maidalun/bce-reranker-base_v1.git
    ```

5. **配置文件**
   - 在`config`下创建`config.yaml`，并在 `config/config.yaml` 中配置 Neo4j 数据库连接信息、OpenAI API 密钥和 ZhipuAI API 密钥。

   - 打开 `config/config.yaml` 文件，配置以下信息：
     ```yaml
     neo4j_uri: "bolt://localhost:7687"
     neo4j_username: "neo4j"
     neo4j_password: "12345678"

     papers_path: "../papers/KGLLM"

     openai_api_key: "Your OpenAI API Key"
     openai_base_url: "Your OpenAI Base Url"

     zhipuai_api_key: "Your ZhipuAI API Key"

     device: "cuda/cpu/mps/..."
     ```

6. **运行项目**
    - 一键生成文献综述知识图谱
        ```bash
        cd src
        python main.py
        ```
    
    - 交互页面
        ```bash
        cd src
        streamlit run web.py
        ```

#### 注意事项

- 确保 Neo4j 数据库已经启动并运行。
- 配置文件中的 API 密钥需要替换为你自己的密钥。
- 如果遇到问题，可以参考项目中的 `README.md` 文件或者提交 Issue。

### 参考链接

- [Neo4j 数据库安装指南](https://neo4j.com/docs/operations-manual/current/installation/)
- [OpenAI API 密钥申请](https://beta.openai.com/docs/developer-quickstart/your-api-keys)
- [ZhipuAI API 密钥申请](https://open.bigmodel.cn/dev/account)

### 许可协议

本项目采用 [MIT 许可](LICENSE) 进行许可。

希望这些步骤能帮助你成功部署 SurveyKG 项目！如果有任何问题，请随时联系我们[huangkywork@163.com](mailto:huangkywork@163.com)。