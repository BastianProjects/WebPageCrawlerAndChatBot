STEPS: 

python -m venv venv


Set-ExecutionPolicy Unrestricted -Scope Process

.\venv\Scripts\activate

pip install -r .\requirements.txt

streamlit run .\PDFEnhancedChatBot.py
Project Description: AI Web Page Crawler Chatbot
The AI Web Page Crawler Chatbot is a Python-based application built with Streamlit that allows users to extract, index, and query information from web pages. It leverages advanced natural language processing (NLP) and vector search technologies to provide concise and accurate answers to user queries based on the content of indexed web pages.
Key Features:
1.	Web Page Crawling:
•	Uses SeleniumURLLoader to load and extract content from user-provided web page URLs.
2.	Text Splitting:
•	Splits large documents into manageable chunks using RecursiveCharacterTextSplitter for efficient indexing and retrieval.
3.	Vector Database:
•	Stores document embeddings in a persistent vector database (Chroma) for fast similarity-based searches.
4.	Embeddings and Language Models:
•	Generates embeddings using the OllamaEmbeddings model and answers user queries with the OllamaLLM language model.
5.	Interactive Chat Interface:
•	Provides a user-friendly chat interface where users can ask questions about the indexed content and receive concise, context-aware answers.
6.	Indexed URL Management:
•	Displays a list of all indexed URLs in the sidebar and allows filtering of answers by specific URLs.
7.	Database Management:
•	Includes a feature to clear the vector database for re-indexing or development purposes.
Use Cases:
•	Quickly extract and query information from web pages.
•	Build a knowledge base from multiple web sources.
•	Enable efficient document search and retrieval for research or business purposes.
This project is ideal for developers, researchers, and businesses looking to automate web content analysis and provide intelligent, context-aware answers to user queries.
