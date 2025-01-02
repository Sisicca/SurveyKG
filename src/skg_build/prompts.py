# Emphasis
EMPHASIS_SYSTEM = """As a professional assistant of science research, you will provide accurate and professional answers to users.
Please ensure that your answer is highly concise."""
EMPHASIS_USER = """Based on the following paper abstract, please summarize the focus(emphasis) of the paper in 10 words or fewer without outputting any additional content: {text}"""
EMPHASIS_ASSISTANT = """how several different technologies(Knowledge Graph and Large Language Model) interact, influence each other, and complement each other's strengths."""

# Extract
EMPHASIS_PROMPT = """As a professional assistant of science research, you will provide accurate and professional answers to users.
Focus on {emphasis} if this paper mentioned it.
"""

PROBLEM_PROMPT = """Abstract the main research question of this paper in 10 words or fewer.
Then, summarize the main research question of this paper in detail.
Please return your answer in the following JSON format, don't output any extra text:
{
    "name": "Research question summarized in 10 words or fewer",
    "description": "Detailed explanation of the research problem"
}"""

METHOD_PROMPT = """
- Requirement -
1. Focus on describing the specific steps, components, and operating mechanisms of the methodology/framework.
2. The extracted content should be a purely methodological or theoretical description, and should not contain content such as research motivation, contribution statements, limitation discussions, or evaluation of results.
3. Preference is given to paragraphs that contain technical details, algorithmic descriptions, mathematical formulas, or flowcharts.
4. If there are multiple related paragraphs, extract them in the order in which they appear in the paper, and indicate the logical relationship between the paragraphs.
5. For long method descriptions, summarize key points appropriately, but make sure you don't miss important details.
6. If there are relevant chapters in the paper that are clearly marked with "methods", "models", "frameworks", etc., please give priority to extracting content from these sections.
7. Exclude content such as experimental setup, data processing, etc., that does not directly describe the core method/framework.
8. Please remember when describing research questions and methods, attach all the original text you used.
- Target -
Based on the requirements above, now abstract and detailedly describe all technique methods and all the techniques used by the method in this paper, and ensure that you contain every technical term and method in this paper.
You can narrate it in several points or by some steps. Please make sure using and explaining Technical Terminology as possible.
Please return your answer in the following JSON format, don't output any extra text:
[
    {
        "name": "Method 1",
        "description": "Detailed description and steps of method 1. ",
    },
    {
        "name": "Method 2",
        "description": "Detailed description and steps of method 2",
    },
    ... (more methods)
]"""

TERMINOLOGY_PROMPT = """Detailedly Describe the most important 10 or less technical terminology in this paper.
Please return your answer in the following JSON format, don't output any extra text:
[
        {"name": "Term 1", "description": "Definition of Term 1"},
        {"name": "Term 2", "description": "Definition of Term 2"},
        ... (more terminologies)
]"""

DATASET_PROMPT = """Find the data sources and describe its effect if the paper used.
Please return your answer in the following JSON format, don't output any extra text:
[
            {"name": "Dataset 1", "description": "Usage of Dataset 1"},
            {"name": "Dataset 2", "description": "Usage of Dataset 2"},
            ... (more datasets)
]"""

# Summary
SUMMARIZE_EMPHASIS = """how several different technologies(Knowledge Graph and Large Language Model) interact, influence each other, and complement each other's strengths."""

SUMMARIZE_SYSTEM = """As a professional assistant of science research, you will provide accurate and professional answers to users.
Please make sure using and explaining Technical Terminology as possible."""

SUMMARIZE_INFO_USER = """
- Target and Process -
To summarize the following paper, focusing on the research questions and technique methods of the paper, you need to think as the following steps:
Step 1 Abstract the research field of this paper. As an expert in this field, you will spare no effort to help newcomers understand the paper from scratch to a thorough understanding of the paper, and master the research questions, ideas, methods, and concepts of the paper.
Step 2. Find the data sources and describe its effect if the paper used.
Step 3. Detailedly Describe all the technical terminology in this paper which you didn't describe in the above process.
Step 4. Abstract the main research question of this paper in 10 words or fewer.
Step 5. Summarize the main research question of this paper in detail.

- Requirements -
The specific requirements are as follows:
Focus on {emphasis} if this paper mentioned it.

- Format -
Please output the above answer in the format below without losing any information:
Data Source:
- ...(Name of dataset 1): Used for ...
- ...(Name of dataset 2): Used for ...
...

Technical terminology:
...(Technical terminologies and thier descriptions)

Research Question: ...(Summary of the research question in 15 words or fewer)
Detailed description of the research question: ...(Description of the research question in detail or in process)

- Paper -
Here is the paper:
###
{text}
###

Let's think as the above 5-step process and answer in the above format:
"""

SUMMARIZE_INFO_ASSISTANT = """
Data Source:
- Family: Used for medical image analysis
- WN18RR: Used for knowledge graph construction
- FB15K-237: Used for knowledge graph construction
- YAGO3-10: Used for knowledge graph construction

Technical terminology:
- **Transfer Learning**: A machine learning approach where knowledge from one domain or task is applied to another related task, enhancing performance especially in situations with limited data availability in the target domain.
- **Knowledge Distillation**: A model compression technique in which a larger, complex model (teacher) imparts its learned knowledge to a smaller, more efficient model (student), maintaining high accuracy with reduced computational resources.
- **Gradient Descent**: An optimization algorithm used to minimize a model's loss function by iteratively adjusting the model parameters in the opposite direction of the gradient to find the optimal solution.
- **Cross-Entropy Loss**: A loss function commonly used in classification tasks, measuring the discrepancy between the true label and predicted probabilities, thus guiding the model’s optimization process.
- **Overfitting**: A phenomenon where a machine learning model learns patterns from the training data too specifically, resulting in poor generalization and performance on new, unseen data.

Research Question: Classification of medical images
Detailed description of the research question: In the classification task, the training of deep learning models faces the problem of insufficient data due to the fact that medical imaging data is usually scarce. Transfer learning enables researchers to efficiently extract features by utilizing pre-trained models on large-scale datasets, thereby improving the accuracy of classification and the generalization ability of models.
"""

SUMMARIZE_CONTENT_USER = """
- Target and Process -
To summarize the following paper, focusing on the research questions and technique methods of the paper, you need to think as the following steps:
Step 1 Abstract the research field of this paper. As an expert in this field, you will spare no effort to help newcomers understand the paper from scratch to a thorough understanding of the paper, and master the research questions, ideas, methods, and concepts of the paper.
Step 2. Abstract and detailedly describe all technique methods and all the techniques used by the method in this paper then attach the original text you used after method, and ensure that you contain every technical term and method in this paper. You can narrate it in several points or by some steps.

- Requirements -
The specific requirements are as follows:
1. Focus on {emphasis} if this paper mentioned it.
2. Focus on describing the specific steps, components, and operating mechanisms of the methodology/framework.
3. The extracted content should be a purely methodological or theoretical description, and should not contain content such as research motivation, contribution statements, limitation discussions, or evaluation of results.
4. Preference is given to paragraphs that contain technical details, algorithmic descriptions, mathematical formulas, or flowcharts.
5. If there are multiple related paragraphs, extract them in the order in which they appear in the paper, and indicate the logical relationship between the paragraphs.
6. For long method descriptions, summarize key points appropriately, but make sure you don't miss important details.
7. If there are relevant chapters in the paper that are clearly marked with "methods", "models", "frameworks", etc., please give priority to extracting content from these sections.
8. Exclude content such as experimental setup, data processing, etc., that does not directly describe the core method/framework.
9. Please remember when describing research questions and methods, attach all the original text you used.
10. You don't need to output the following format first.
11. Please output the above answer in the format below without losing any information:

- Format -
Method 1: ...(Summary of the method 1 in 15 words or fewer)
Detailed description of method 1:
...(Description of the method 1 in detail)
Refer to the original text:
...(Original text)

Method 2: ...(Summary of the method 2 in 15 words or fewer)
Detailed description of method 2:
...(Description of the method 2 in detail)
Refer to the original text:
...(Original text)

......(Method 3 and so on)

- Paper -
Here is the paper:
###
{text}
###

Let's think as the above 2-step process and answer in the above format:
"""


SUMMARIZE_CONTENT_ASSISTANT = """
Method 1: Transfer Learning
Detailed description of method 1:
The core idea of transfer learning is to apply what is learned on one task to another related but different task. In medical image analysis, due to the high cost and time to obtain high-quality annotated data, transfer learning improves performance in small-sample medical imaging tasks by borrowing deep learning models pre-trained on large-scale datasets such as ImageNet. The use process of research method 1:
    1. Choose a pre-trained model: Researchers typically choose deep convolutional neural networks (CNNs) trained on large-scale image datasets, such as VGG, ResNet, or Inception. These models have learned rich image features, such as edges, textures, and shapes, that are equally important for medical image analysis tasks.
    2. Feature extraction: In the first step of transfer learning, a pretrained model is used as a feature extractor in the paper, using its first few layers (usually a convolutional layer) to extract low-level features in medical images. These features are able to capture patterns that are common in medical imaging, such as tissue structures and abnormalities.
    3. Fine-tune the model: After the feature extraction is completed, the researcher will fine-tune the model. Specifically, the last few layers of the pretrained model are typically replaced with output layers that are appropriate for the task at hand (for example, binary or multi-classification layers). Then, the entire model was retrained with a small-scale medical imaging dataset, especially for more fine-tuning of the newly added output layers. In this way, the model is able to learn features that are more suitable for specific medical imaging tasks.
    4. Training process: In the training process, a small learning rate is usually used to avoid drastically changing the weight of the pre-trained layer, so as to maintain the features learned by the pre-trained model. In addition, researchers will use data augmentation techniques (such as rotation, scaling, and flipping) to expand the training set and improve the generalization ability of the model.
    5. Evaluation and validation: After training, the researchers in the paper evaluate the fine-tuned model using an independent test set. Evaluation metrics include accuracy, sensitivity, specificity, and AUC (area under the curve) to provide a comprehensive picture of the model's performance. Validate the effectiveness of transfer learning by comparing it with other methods, such as a de novo trained model.
    6. Summary: Transfer learning provides an effective solution for medical image analysis, which can achieve high-precision classification and segmentation in the case of limited data. By borrowing the knowledge of the pre-trained model, the researchers not only accelerated the training process of the model, but also significantly improved the performance of the model on specific tasks. This approach has shown great potential in practical applications to support medical diagnosis and treatment decisions.
Refer to the original text:
    1. Choose a Pre-trained Model: The first step in transfer learning is to select an appropriate pre-trained model. In medical image analysis, deep convolutional neural networks (CNNs) such as VGG, ResNet, or Inception are commonly used. These models have been pre-trained on large-scale image datasets like ImageNet, enabling them to learn useful image features such as edges, textures, and shapes. These learned features are generally transferable to other tasks, including medical image analysis, where similar patterns like tissue structures and abnormalities are often present.
    2. Feature Extraction: Once a pre-trained model is selected, it is used as a feature extractor. In this step, the model's initial layers, typically the convolutional layers, are used to extract low-level features from the medical images. These low-level features capture general image patterns, which are often shared across different image domains. For medical images, these features might include textures or edges that help in identifying key structures, such as tumors or organs, making them highly relevant for tasks like disease detection or tissue segmentation.
    3. Fine-tuning the Model: After the pre-trained model has been used to extract features, the next step is fine-tuning. In this process, the model’s final layers (usually the fully connected layers) are replaced with new layers suited for the specific task at hand, such as binary or multi-class classification. The entire model is then retrained on a smaller medical dataset to adjust the new output layers. During fine-tuning, the model retains its ability to recognize general image features learned from the large-scale dataset while adapting its parameters to more closely match the unique characteristics of the medical data.
    4. Training Process: During the training phase, a small learning rate is typically used to ensure that the pre-trained layers are not drastically altered. This helps preserve the knowledge learned from the large dataset. The model is gradually adapted to the new task without losing the valuable features it has already learned. To enhance the model’s generalization ability and prevent overfitting, data augmentation techniques such as rotation, scaling, and flipping are applied to the medical image dataset. These techniques artificially expand the dataset, allowing the model to become more robust and capable of handling various image orientations and transformations.
    5. Evaluation and Validation: Once the model is trained, it is evaluated using an independent test set that was not used during training. The performance of the fine-tuned model is assessed using a range of metrics, including accuracy, sensitivity, specificity, and the area under the curve (AUC). These metrics provide a comprehensive understanding of how well the model performs, particularly in distinguishing between different classes (e.g., healthy vs. diseased). The effectiveness of transfer learning is further validated by comparing the performance of the transfer-learned model to a model trained from scratch (i.e., de novo training) on the same medical dataset.
    6. Transfer learning offers an effective solution for medical image analysis, especially in cases where annotated medical datasets are limited. By leveraging the knowledge learned from large-scale datasets, transfer learning accelerates the training process and improves the performance of the model on specific tasks. This approach enables high-precision classification and segmentation, even with smaller amounts of medical data. In practical applications, transfer learning can significantly enhance the accuracy of medical diagnoses and support clinical decision-making, demonstrating its potential for widespread use in healthcare.

Method 2: Entity Relationship Modeling
Detailed description of Method 2:
Entity relationship modeling is one of the core methods in knowledge graph construction, and its main purpose is to describe the connections and interactions between data by establishing nodes (entities) and edges (relationships). In practice, entities represent objects in a knowledge graph, such as people, places, or concepts, while relationships represent associations between entities, such as "located", "owned", or "belonged". This modeling approach can help us better organize, understand, and mine the hidden knowledge in our data. How to use Research Method 1:
    1. Identify entities and relationships: The researcher first identifies the key entities in the knowledge graph and the relationships between them. These entities and relationships are often derived from domain knowledge or prior knowledge and are screened and validated through steps such as data preprocessing and labeling.
    2. Data extraction and cleansing: Extract entity and relational data from different data sources, such as text, databases, or websites. The data cleansing step is especially important because the data can be redundant or noisy, ensuring the accuracy of the knowledge graph by deduplication and filtering out invalid data.
    3. Triplet construction: Each pair of entities and relationships is formed into a triplet (Entity 1 Relationship Entity 2) to represent a basic unit in the knowledge graph. The triplet format facilitates subsequent graph database storage and query.
    4. When a knowledge graph comes from multiple sources, entities in different sources may appear in different names or expressions. Researchers use disambiguation techniques, such as rule-based matching or machine learning models, to merge identical entities and ensure consistency.
    5. Once built, the knowledge graph is usually stored in a graph database to support efficient query and analysis. Graph databases use graph structures to store entities and relationships, which facilitates fast retrieval and relational inference.
    6. Entity relationship modeling plays a key role in the knowledge graph. By systematically structuring and managing entities and relationships, knowledge graphs can support a variety of applications, such as intelligent recommendations, question answering systems, and semantic search, to help users quickly find useful information in complex data.
Refer to the original text:
    1. The first step in entity relationship modeling involves identifying the key entities and relationships in the domain. Entities represent the objects or concepts that are important in the knowledge graph, such as people, places, products, or events. Relationships, on the other hand, describe the interactions or associations between these entities. Examples of relationships include "located in," "owned by," or "related to." Researchers use domain knowledge or prior information to identify relevant entities and relationships, and then validate and refine these through data preprocessing and labeling processes.
    2. Once entities and relationships are identified, data is extracted from various sources, such as text documents, databases, or websites. This data extraction step is crucial to gather all the necessary information to build the knowledge graph. Afterward, data cleansing is performed to improve the quality of the data. This involves eliminating redundant or noisy data, removing inconsistencies, and ensuring that the information is accurate. Data cleansing steps like deduplication and filtering invalid data ensure that the knowledge graph is built on reliable, clean data.
    3. In this step, entities and their corresponding relationships are structured into triplets. Each triplet represents a basic unit of the knowledge graph and is formatted as (Entity 1, Relationship, Entity 2). For example, a triplet could be ("John", "works for", "Company A"). This triplet format makes it easier to represent relationships and store them in a graph database, as well as facilitates querying and analysis. Triplets form the core structure of the knowledge graph, and organizing data in this way allows for efficient knowledge representation and retrieval.
    4. When constructing a knowledge graph from multiple sources, it's common for the same entity to appear under different names or expressions. For example, "IBM" and "International Business Machines" refer to the same entity. To ensure consistency across the knowledge graph, knowledge fusion and disambiguation techniques are applied. These techniques, such as rule-based matching or machine learning models, help to identify and merge identical entities, removing ambiguity and ensuring that the knowledge graph accurately represents the real-world entities and their relationships.
    5. After constructing the knowledge graph, it is typically stored in a graph database. A graph database is optimized for storing and querying graph-structured data, making it ideal for handling the entities and relationships in a knowledge graph. Graph databases facilitate efficient querying and relational inference, allowing users to easily retrieve related information and perform complex queries. This storage format supports a range of operations, such as traversing relationships between entities, performing pattern matching, and running graph-based algorithms, all of which enhance the usability of the knowledge graph.
    6. Entity relationship modeling plays a fundamental role in the creation of knowledge graphs. By systematically identifying and structuring entities and relationships, researchers can build a knowledge graph that represents complex data in an organized and easily queryable format. Knowledge graphs are valuable for a wide range of applications, including intelligent recommendation systems, question-answering systems, and semantic search. These applications enable users to efficiently find relevant information from large datasets, providing insights that are difficult to extract from raw data alone. Through the careful application of entity relationship modeling, knowledge graphs become powerful tools for managing and mining knowledge from complex, interconnected data sources.

Method 3: Semantic Relationship Mapping
Detailed description of method 3:
Semantic relationship mapping is a fundamental technique in knowledge graph construction that focuses on understanding and encoding the meaning behind connections between entities. This method allows the knowledge graph to capture more nuanced insights about how different entities interact, enriching the graph’s ability to support advanced reasoning and complex queries. Process for research method 1: 
    1. Identify Entities and Relationships: Researchers begin by defining the primary entities (such as people, places, events) and semantic relationships (e.g., “is friend of,” “works at,” “occurred during”) based on domain knowledge, ensuring all connections are contextually meaningful.
    2. Data Extraction and Processing: Data is extracted from structured and unstructured sources (like databases, documents, and web data). The extracted information is then processed to remove redundancy and noise, ensuring accuracy.
    3. Triple Construction: The extracted entities and relationships are organized into triples (Entity1 - Relationship - Entity2) to represent knowledge in a structured form. This representation enables efficient storage and retrieval.
    4. Disambiguation and Consistency Checking: Since entities might have different names or variations, researchers use techniques such as natural language processing (NLP) and machine learning to disambiguate and ensure consistency across the knowledge graph.
    5. Graph Storage and Query Optimization: The completed knowledge graph is stored in a graph database, where entities and relationships can be indexed to support high-speed queries and complex reasoning, enabling deeper insights from interconnected data.
    6. Summary: Semantic relationship mapping enhances a knowledge graph's ability to provide meaningful, context-rich information by accurately representing entity relationships. This method supports diverse applications like personalized recommendations, intelligent search, and predictive modeling, allowing users to uncover insights that go beyond surface-level data connections.
Refer to the original text:
    1. The first step in semantic relationship mapping is to define the primary entities and their relationships within a given domain. Entities might include people, places, events, or objects, while relationships capture the interactions between these entities. For example, relationships could be "is friend of," "works at," "occurred during," or "located in." These relationships provide important context and meaning to the knowledge graph, ensuring that the graph represents not just raw data but also the underlying significance of how different entities are connected. Researchers carefully define these entities and relationships based on domain expertise to ensure that all connections are meaningful and relevant.
    2. Once entities and relationships are identified, data is extracted from various structured and unstructured sources, such as databases, documents, and websites. This data extraction process may involve parsing large amounts of text or accessing external data sources. After extraction, the data undergoes a processing phase, where redundant or noisy information is removed. This is crucial to ensure that the data used to build the knowledge graph is clean and accurate, which enhances the quality of the graph and its ability to support meaningful queries and insights.
    3. After processing the data, the next step is to organize the extracted entities and relationships into triples. Each triple consists of three components: (Entity1 - Relationship - Entity2). For example, in a graph of social connections, a triple might be ("John," "is friend of," "Alice"). This triplet format captures the relationship between two entities in a structured way, enabling efficient storage and retrieval in a graph database. The use of triples ensures that the data is represented in a consistent, easy-to-query format that supports complex analysis.
    4. In many real-world scenarios, entities can appear in multiple forms or under different names. For instance, "IBM" and "International Business Machines" refer to the same entity, but may be written differently in various data sources. To resolve such issues, researchers use techniques such as natural language processing (NLP) and machine learning to disambiguate entities. This ensures that different names referring to the same entity are correctly identified and merged. Consistency checking is also performed to make sure that the relationships and entity names remain uniform across the entire knowledge graph, leading to a more accurate and cohesive representation of knowledge.
    5. Once the knowledge graph is constructed, it is stored in a graph database. Graph databases are particularly well-suited for storing entity-relationship data because they allow for efficient querying of relationships and entities. The entities and relationships in the graph are indexed to optimize the retrieval process, enabling high-speed queries and advanced reasoning capabilities. This optimization is essential for applications that require real-time data analysis or complex graph traversal. Through these optimizations, the knowledge graph can support deeper insights and allow users to explore interconnected data in meaningful ways.
    6. Semantic relationship mapping is an essential technique for enhancing the richness and depth of a knowledge graph. By capturing not only the entities but also the contextual meaning behind their relationships, this method allows the knowledge graph to provide a more accurate and nuanced understanding of data. It supports various applications, such as personalized recommendations, intelligent search engines, and predictive modeling. These applications can leverage the graph’s ability to uncover insights beyond surface-level connections, providing valuable intelligence for decision-making and analysis. Through semantic relationship mapping, a knowledge graph becomes a powerful tool for extracting meaningful information from complex, interconnected data sources.
......
"""


# Direction
DIRECTION_SYSTEM = """As a professional assistant of science research with extensive knowledge who is good at summarizing, you will provide accurate and professional answers to users.
"""

DIRECTION_USER = """
- Target and Process -
To summarize and extract the main research direction (primary direction) and its specific sub directions (secondary direction) from given research questions(which are extracted from professional papers),  you need to think as the following steps:
1. Read and understand each research question.
2. Cluster, extract, and summarize the main research directions (primary directions) in 10 words or fewer for each direction.
3. Identify specific research sub directions (secondary directions) in each primary direction in 10 words or fewer for each direction.
4. Provide detailed descriptions for each direction and clarify the direction to which each paper belongs.

- Requirements -
1. Make sure that each question is categorized in one secondary direction.
2. Each question can be categorized in multiple secondary directions
3. Ensure clear differentiation between each direction.
4. Ensure that the research tasks in the secondary direction belong to the primary direction.
5. Ensure that the classification is correct and accurate.
6. Do not classify too few so that there is no distinction, nor classify too many so that similar categories are formed.

- Format -
To output without losing any information and clearly indicate which direction each paper belongs to, please output the above answer in the format below strictly:
Primary Direction 1: ...(Name the primary direction 1 in 10 words or fewer)
Detailed description of primary direction 1: ...(Description of the primary direction 1 in detail)

Secondary Direction 1.1: ...(Name the secondary direction 1.1 in 10 words or fewer)
Detailed description of secondary direction 1.1: ...(Description of the secondary direction 1.1 in detail)
Number: [List of problem numbers belonging to this secondary direction]

Secondary Direction 1.2: ...(Name the secondary direction 1.2 in 10 words or fewer)
Detailed description of secondary direction 1.2: ...(Description of the secondary direction 1.2 in detail)
Number: [List of problem numbers belonging to this secondary direction]

......(Secondary Direction 1.3 and so on)

Primary Direction 2: ...(Name the primary direction 2 in 10 words or fewer)
Detailed description of primary direction 2: ...(Description of the primary direction 2 in detail)

Secondary Direction 2.1: ...(Name the secondary direction 2.1 in 10 words or fewer)
Detailed description of secondary direction 2.1: ...(Description of the secondary direction 2.1 in detail)
Number: [List of problem numbers belonging to this secondary direction]

......(Secondary Direction 2.2 and so on)

......(More Primary Direction and Secondary Direction)

- Research Problems to analyze-
{problem_string}

Let's think as the above 4-step process and answer in the above format:
"""

DIRECTION_ASSISTANT = """
Primary Direction 1: Large-Scale Language Models and Knowledge Augmentation
Description of Primary Direction 1: Exploring how to enhance language models' performance through knowledge injection, representation, and reasoning to improve contextual understanding, reasoning performance, and knowledge coverage.

Secondary Direction 1.1: Knowledge Injection for Language Models
Description of Secondary Direction 1.1: Investigating methods to integrate external knowledge resources (e.g., encyclopedic data, knowledge bases) into language models to enhance knowledge density and accuracy.
Number: [1, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40]

Secondary Direction 1.2: Knowledge Representation and Reasoning in Pre-trained Models
Description of Secondary Direction 1.2: Examining the applications of pre-trained language models in knowledge representation and reasoning, including effective use of embedded knowledge for complex problem-solving.
Number: [2, 4, 8, 11, 13, 18, 23, 28, 32, 38, 42]

Secondary Direction 1.3: Integration of Knowledge Graphs and Language Models
Description of Secondary Direction 1.3: Combining knowledge graphs with language models to enhance reasoning through semantic relationships, achieving more precise information extraction and knowledge expression.
Number: [9, 12, 16, 22, 27, 34, 36, 41, 45, 47, 49]

......

Primary Direction 2: Reinforcement Learning and Decision Optimization
Description of Primary Direction 2: Applying reinforcement learning combined with deep learning in diverse tasks to achieve automated decision optimization and strategy generation, especially in uncertain and complex scenarios.

Secondary Direction 2.1: Applications of Deep Reinforcement Learning in Multi-agent Systems
Description of Secondary Direction 2.1: Researching reinforcement learning applications in multi-agent systems to address optimization problems in agent collaboration, gaming, and competition tasks.
Number: [5, 10, 17, 21, 26, 33, 37, 40, 43, 48, 51]

Secondary Direction 2.2: Combination of Reinforcement Learning and Meta-Learning
Description of Secondary Direction 2.2: Combining meta-learning with reinforcement learning to accelerate the learning process for new tasks, enhancing model adaptability and generalization in new environments.
Number: [3, 9, 15, 20, 23, 28, 31, 35, 39, 44, 52]

......
"""





# Domain
DOMAIN_SYSTEM ="""As a professional assistant of science research with extensive knowledge who is good at summarizing, you will provide accurate and professional answers to users.
"""

DOMAIN_USER = """
- Target -
Please name the research field based on the content and characteristics of the following research directions.
Then describe the domain in detail.
Note that only output the name you have given to the research field, do not output any additional content.

- Format -
Domain: ...(Name the domain of research directions in 10 words or fewer)
Detailed description of domain: ...(Describe the domain in detail)

- Research directions in the field to name -
{directions}
Please name the field of the above research direction in 10 words or fewer and then describe it in detail:
"""

DOMAIN_ASSISTANT = """
Domain: Biology and AI
Detailed description of domain: Biology Integrated with AI combines artificial intelligence and biological sciences to enhance research, analysis, and innovation in fields like genomics, proteomics, and drug discovery. AI-driven techniques, such as machine learning and computer vision, are used to analyze complex biological data, predict molecular interactions, and accelerate the discovery of treatments. This integration enables more efficient data interpretation, personalized medicine, and insights into biological processes that drive advancements in healthcare and biotechnology.
"""





# Technical Roadmap
TECHNOLOGY_SYSTEM = """As a professional assistant of science research with extensive knowledge who is good at summarizing, you will provide accurate and professional answers to users.
"""

TECHNOLOGY_USER = """
- Target and Process -
To summarize and extract the main Technical Route (Technology) from given research methods(which are extracted from professional papers),  you need to think as the following steps:
1. Read and understand each research method.
2. While focus on the commonalities and similarities between different methods, cluster, extract and summarize the main Technical Routes (Technology), which exist in these method, in 15 words or fewer for each route.
3. Provide detailed descriptions for each technical route, highlight its core features and innovations.
4. Clarify the technical route to which each paper belongs.

- Requirements -
1. Only care about the main core important technology classification, not subdivision.
2. Make sure that each method is categorized in at least one technical route.
3. Each method can be categorized in multiple technical routes
4. Ensure clear differentiation between each technical route.
5. Ensure that the classification is correct and accurate, make sure that each technical route is clearly distinguishable, but also try to classify papers using similar methods into the same technical route.
6. Do not classify too few so that there is no distinction, nor classify too many so that similar categories are formed.

- Format -
To output without losing any information and clearly indicate which direction each paper belongs to, please output the above answer in the format below strictly:
Technical Route 1: ...(name of the technical route in 15 words or fewer)
Detailed description of technical route 1: ...(Describe the technical route in detail)
Number: [List of method numbers belonging to this technical route]

Technical Route 2: ...(name of the technical route in 15 words or fewer)
Detailed description of technical route 2: ...(Describe the technical route in detail)
Number: [List of method numbers belonging to this technical route]

......(More Technical Route)

- Research Methods to analyze-
{method_string}

Let's think as the above 4-step process and answer in the above format:
"""

TECHNOLOGY_ASSISTANT = """
Technical route 1: Named entity recognition is performed using large models
Detailed description of technical route 1: Named Entity Recognition with Large Models utilizes advanced, large-scale language models to identify and classify entities—such as names of people, places, organizations, and dates—in unstructured text. Leveraging large models, such as transformers, improves the accuracy and context-awareness of NER systems, allowing for better handling of complex language, rare entities, and nuanced text. This approach is widely used in applications like information extraction, data mining, and automated knowledge graph construction.
Number: [1, 2, 5]

Technical route 2: Knowledge distillation is used to extract knowledge graphs from large models
Detailed description of technical route 2: Knowledge Distillation for Knowledge Graph Extraction applies knowledge distillation techniques to transfer insights from large language models into more efficient representations, such as knowledge graphs. This process involves "distilling" the complex relationships and factual knowledge embedded in large models into structured graphs, capturing essential entities and their relationships. This approach enables efficient storage, faster retrieval, and improved interpretability, making it valuable for applications requiring structured knowledge, like recommendation systems, semantic search, and question answering.
Number: [3, 4, 6, 7]

......(and so on)
"""

SOURCE_PROMPT = """
- SOURCE DOCUMENTS TO ANSWER QUESTION -
You have to use documents below to complete the tasks of user.
{source}
"""