# PDFChat

PDFChat is a Streamlit-based web application that allows users to upload PDF documents and interactively ask questions about their content. The app leverages document chunking, vector embeddings, and large language models to provide detailed, context-aware answers.

## Features

- Upload PDF files and process them into searchable document chunks.
- Store document embeddings in a persistent ChromaDB vector database.
- Query the database with natural language questions.
- Retrieve and re-rank relevant document chunks using cross-encoder models.
- Generate comprehensive answers using an LLM (Ollama/Gemma3).
- View retrieved documents and most relevant document IDs.

## Installation

1. **Clone the repository:**

   ```sh
   git clone http://github.com/Avaneesh-Chopdekar/pdfchat
   cd pdfchat
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

   Or use [pyproject.toml](pyproject.toml) with your preferred tool (e.g., `pip`, `poetry`).

3. **Start Ollama embedding server:**
   - Ensure Ollama is running locally on port 11434.

## Usage

1. **Run the Streamlit app:**

   ```sh
   streamlit run app.py
   ```

2. **Upload a PDF:**

   - Use the sidebar to upload your PDF and process it.

3. **Ask questions:**
   - Enter your question in the main interface and click "Ask".
   - View the AI-generated answer and explore retrieved document chunks.

## Project Structure

- [`app.py`](app.py): Main Streamlit application.
- [`prompt.py`](prompt.py): System prompt for LLM context.
- [`pdf-chat-chroma/`](pdf-chat-chroma): Persistent ChromaDB vector database.
- [`pyproject.toml`](pyproject.toml): Project dependencies.
- [`README.md`](README.md): Project documentation.

## Dependencies

- streamlit
- chromadb
- langchain-community
- ollama
- pymupdf
- sentence-transformers

See [`pyproject.toml`](pyproject.toml) for exact versions.

## Notes

- The app requires Ollama to be running locally for embedding and chat.
- Uploaded PDFs are processed and stored as vector embeddings for efficient retrieval.
- All answers are generated based solely on the provided PDF context.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.
