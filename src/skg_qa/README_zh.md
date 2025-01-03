# skg_qa 模块

- [English Version](README.md)
- [中文版](README_zh.md)

## 概述

`skg_qa` 模块为 SurveyKG 提供了复杂的问答功能。通过利用语言模型和 Neo4j 的图数据库，该模块允许用户对构建的知识图谱执行自然语言查询，检索精确且上下文相关的信息。

## 目录

- [skg\_qa 模块](#skg_qa-模块)
  - [概述](#概述)
  - [目录](#目录)
  - [功能](#功能)

## 功能

- **基于Cypher的问答 (CypherQaLLM)**：生成Cypher查询以从Neo4j中检索结构化数据，为用户问题提供精确答案。
- **思维导图问答 (MindMapQaLLM)**：结合语义检索和子图可视化，提供富有上下文的答案。
- **检索增强生成 (RagQaLLM & LightRagQaLLM)**：通过从知识图谱中检索相关文档或上下文，增强答案生成。
- **自定义链**：实现自定义的QA链，以整合多个检索和生成步骤。
- **基于示例的学习**：利用预定义的示例指导语言模型生成准确的查询和响应。
- **错误处理和重试**：强大的机制处理API调用失败，确保可靠性能。