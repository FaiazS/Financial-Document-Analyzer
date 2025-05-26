# Financial Document Analyzer

A visual RAG (Retrieval-Augmented Generation) application for querying and analyzing financial documents like annual reports. Built with LangChain, CrewAI, ChromaDB, and Dash.

## Features

- Upload and process multiple financial documents (PDF, DOCX, TXT)
- Natural language querying of document contents
- Document analysis using AI agents
- Interactive visualization of document insights
- Vector-based semantic search

## Prerequisites

- Python 3.8+
- OpenAI API key (for GPT-3.5-turbo)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd financial-rag-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   Create a `.env` file in the project root and add:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:8050/`

3. Upload financial documents using the upload interface

4. Ask questions about the uploaded documents using natural language

## How It Works

1. **Document Processing**:
   - Documents are loaded and split into chunks
   - Text is embedded using HuggingFace's all-MiniLM-L6-v2 model
   - Embeddings are stored in ChromaDB for efficient retrieval

2. **Query Processing**:
   - User queries are converted to embeddings
   - Relevant document chunks are retrieved using vector similarity
   - A language model generates answers based on the retrieved context

3. **Document Analysis**:
   - CrewAI agents analyze the documents for key insights
   - Results are presented in an easy-to-understand format

## Project Structure

- `app.py`: Main application file with Dash UI and backend logic
- `requirements.txt`: Python dependencies
- `data/`: Directory for uploaded documents and ChromaDB storage
- `.env`: Environment variables (create this file)

## Customization

- To use a different language model, modify the `query_documents` function in `app.py`
- Adjust chunk size and overlap in the `process_uploaded_files` function
- Add more analysis agents in the `analyze_document` function

## License

MIT License - Feel free to use and modify this project for your needs.
