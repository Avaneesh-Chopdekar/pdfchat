import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile


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


if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title="Chat with PDF", page_icon="📄")
        st.header("Chat with PDF")
        uploaded_file = st.file_uploader(
            "🔼 Upload your PDF", type=["pdf"], accept_multiple_files=False
        )

        process_pdf = st.button("⚡ Process PDF")

    if uploaded_file and process_pdf:
        all_splits = process_document(uploaded_file)
        st.write(all_splits)
