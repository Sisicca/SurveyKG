# skg_qa Module

- [English Version](README.md)
- [中文版](README_zh.md)

## Overview

The `skg_qa` module empowers SurveyKG with sophisticated question-answering capabilities. By leveraging language models and Neo4j's graph database, this module allows users to perform natural language queries on the constructed knowledge graph, retrieving precise and contextually relevant information.

## Table of Contents

- [skg\_qa Module](#skg_qa-module)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)

## Features

- **Cypher-Based QA (CypherQaLLM)**: Generates Cypher queries to retrieve structured data from Neo4j, providing precise answers to user questions.
- **MindMap QA (MindMapQaLLM)**: Combines semantic retrieval with subgraph visualizations to offer context-rich answers.
- **Retrieval-Augmented Generation (RagQaLLM & LightRagQaLLM)**: Enhances answer generation by retrieving relevant documents or context from the knowledge graph.
- **Custom Chains**: Implements custom QA chains to integrate multiple retrieval and generation steps.
- **Example-Based Learning**: Utilizes predefined examples to guide the language models in generating accurate queries and responses.
- **Error Handling and Retries**: Robust mechanisms to handle API call failures and ensure reliable performance.