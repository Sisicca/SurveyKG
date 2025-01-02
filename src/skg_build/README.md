# skg_build Module

- [English Version](README.md)
- [中文版](README_zh.md)

## Overview

The `skg_build` module is the backbone of SurveyKG, responsible for processing academic papers and constructing the knowledge graph. It automates the extraction of key information such as research problems, methods, terminologies, and datasets from PDF documents, and organizes this data into a structured format suitable for graph representation.

## Table of Contents

- [skg\_build Module](#skg_build-module)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)

## Features

- **File Processing**: Recursively search and process PDF files from specified directories.
- **Abstract Extraction**: Automatically extract abstracts from PDF documents using regex patterns.
- **Text Splitting**: Divide large text content into manageable chunks for efficient processing.
- **Information Extraction**: Utilize AI models to extract research problems, methods, terminologies, and datasets from abstracts.
- **Vector Database Construction**: Build a FAISS-based vector database to support semantic retrieval.
- **Knowledge Graph Population**: Create nodes and relationships in the Neo4j graph database based on extracted information.
- **Error Handling**: Implement retry mechanisms to handle API call failures and other exceptions.
- **Parallel Processing**: Leverage multi-threading to expedite the extraction and construction processes.
- **Data Persistence**: Save extracted data into JSON files for future reference and processing.

