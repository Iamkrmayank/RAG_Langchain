import os
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
import streamlit as st

class RAGSystem:
    def __init__(self):
        self.vector_store = {}
        self.index = None
        self.embedding_size = 768  # Adjust this based on your embedding model size
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Sentence transformer model
        
        # Use the OpenAI key from Streamlit secrets
        self.openai_api_key = st.secrets['OPENAI_API_KEY']
        openai.api_key = self.openai_api_key

    def initialize_system(self):
        """Initialize the vector store and FAISS index."""
        vector_store_path = 'vector_store/index.faiss'

        if not os.path.exists(vector_store_path):
            # FAISS index does not exist, create a new one
            self.index = faiss.IndexFlatL2(self.embedding_size)  # L2 distance index
            self.vector_store = {}

            # Ensure the directory exists
            os.makedirs('vector_store', exist_ok=True)

            # Save an empty index
            faiss.write_index(self.index, vector_store_path)
        else:
            # If index exists, load it
            self.index = faiss.read_index(vector_store_path)
            self.vector_store = self.load_documents()  # Load documents if necessary

    def save_vector_store(self):
        """Save the current FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, 'vector_store/index.faiss')

    def load_documents(self):
        """Load stored documents."""
        # Implement document loading if you're saving documents
        return {}

    def query(self, question):
        """Query the vector store with a question and use OpenAI for response."""
        # Ensure the index is initialized
        if self.index is None:
            raise ValueError("FAISS index is not initialized. Please initialize the system first.")

        # Generate the embedding for the question using the model
        question_embedding = self.model.encode([question])[0].astype('float32')

        # Perform search
        _, indices = self.index.search(np.array([question_embedding]), k=5)  # Search for 5 closest documents
        retrieved_docs = [self.vector_store.get(i) for i in indices[0]]  # Retrieve corresponding documents

        # Combine the retrieved documents and form the context for the model
        context = "\n".join(retrieved_docs)

        # Use OpenAI to generate an answer based on the context and the question
        try:
            response = openai.Completion.create(
                model="text-davinci-003",  # You can change this to any model like GPT-4, etc.
                prompt=f"Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {question}\nAnswer:",
                max_tokens=200
            )
            answer = response.choices[0].text.strip()
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def add_document(self, doc, embedding):
        """Add a document to the vector store."""
        doc_id = len(self.vector_store)  # Simple unique ID generation
        self.vector_store[doc_id] = doc  # Store document
        self.index.add(np.array([embedding], dtype='float32'))  # Add the embedding to the FAISS index
