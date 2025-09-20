import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile


import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    # Storing uploaded file as temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)  # Delete the temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", "", ".", "?", "!"],
    )

    return text_splitter.split_documents(docs)


def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./pdf-chat-chroma")
    return chroma_client.get_or_create_collection(
        name="pdf-chat",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ollama_ef,
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    st.success("Data added to vector collection successfully")


def query_collection(prompt: str, n_results: int = 10) -> chromadb.QueryResult:
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title="Chat with PDF", page_icon="📄")
        uploaded_file = st.file_uploader(
            "🔼 Upload your PDF", type=["pdf"], accept_multiple_files=False
        )

        process_pdf = st.button("⚡ Process PDF")

        if uploaded_file and process_pdf:
            normalize_uploaded_filename = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_filename)

    st.header("🤖 Chat with PDF")
    prompt = st.text_area("📝 Ask a question about your PDF:")
    ask_btn = st.button("🚀 Ask")

    if prompt and ask_btn:
        results = query_collection(prompt)
        st.write(results)
