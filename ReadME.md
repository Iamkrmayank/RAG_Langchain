# RAG Query System

A Streamlit-based application for querying documents using RAG (Retrieval Augmented Generation).

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-query-system.git
cd rag-query-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload text documents using the sidebar
2. Enter your questions in the main interface
3. Get AI-powered answers based on your documents

## Project Structure

- `app.py`: Streamlit application
- `rag_system.py`: RAG system implementation
- `documents/`: Directory for uploaded documents
- `vector_store/`: Directory for FAISS vector store

## Requirements

- Python 3.8+
- OpenAI API key
- See requirements.txt for full dependencies

## License

MIT License
