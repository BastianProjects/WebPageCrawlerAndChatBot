import streamlit as st
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os
import shutil
import gc

# -- Settings --
CHROMA_DB_DIR = "chroma_DB"
COLLECTION_NAME = "everything"
EMBED_MODEL = "llama3.2"
LLM_MODEL = "gemma3"

# -- Templates --
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# -- Initialize Models --
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = OllamaLLM(model=LLM_MODEL)

# -- Cached Vector Store Instance --
@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

# -- Helper Functions --
def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    if not documents:
        st.error(f"Error: No content loaded from {url}")
    else:
        st.success(f"Successfully loaded {len(documents)} documents from {url}.")
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    
    # Check if chunks are empty
    if not chunks:
        st.error("Error: No valid chunks created.")
    else:
        st.success(f"Successfully created {len(chunks)} chunks.")
    
    return chunks

def index_docs(docs, url):
    store = get_vector_store()
    wrapped = [
        Document(page_content=doc.page_content, metadata={"source_url": url})
        for doc in docs
    ]
    
    # Ensure there is content to add
    if not wrapped:
        st.error("Error: No documents to index.")
    else:
        st.success(f"Indexing {len(wrapped)} documents from {url}.")
    
    store.add_documents(wrapped)
    store.persist()

def retrieve_docs(query, filter_url=None):
    store = get_vector_store()
    if filter_url and filter_url != "All":
        return store.similarity_search(query, k=4, filter={"source_url": filter_url})
    return store.similarity_search(query, k=4)

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context})

# -- Streamlit UI --
st.title("🔎 AI Web Page Crawler")

# -- Clear DB Button --
if st.sidebar.button("🧹 Clear Vector DB"):
    try:
        del st.session_state["vector_store"]
    except KeyError:
        pass
    gc.collect()
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        st.sidebar.success("Vector store cleared.")

# -- Show Indexed URLs in Sidebar --
store = get_vector_store()
try:
    all_docs = store.similarity_search("list", k=100)
    all_urls = sorted(set(doc.metadata.get("source_url") for doc in all_docs if "source_url" in doc.metadata))
    st.sidebar.markdown("### 📚 Indexed URLs:")
    for u in all_urls:
        st.sidebar.write(f"- {u}")
except Exception as e:
    all_urls = []
    st.sidebar.error("Could not list sources.")

# -- Enter URL to Index --
url = st.text_input("Enter a Web Page URL to index:")

if url:
    with st.spinner("🔄 Loading and indexing page..."):
        documents = load_page(url)
        chunks = split_text(documents)
        index_docs(chunks, url)
    st.success("✅ Page indexed!")

# -- URL Filter Dropdown --
selected_url = st.selectbox("Filter answers by URL (or 'All'):", ["All"] + all_urls)

# -- Chat Input --
question = st.chat_input("Ask a question...")

if question:
    st.chat_message("user").write(question)

    with st.spinner("🤖 Thinking..."):
        results = retrieve_docs(question, selected_url)
        context = "\n\n".join([doc.page_content for doc in results])
        answer = answer_question(question, context)

    st.chat_message("assistant").write(answer)

    with st.expander("📄 Retrieved Chunks"):
        for i, doc in enumerate(results, 1):
            st.markdown(
                f"**Chunk {i} (from {doc.metadata.get('source_url')}):**\n\n"
                f"{doc.page_content[:500]}..."
            )
