import os
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

class RAGSystem:
    def __init__(self, documents_dir="./documents"):
        self.documents_dir = documents_dir
        self.embeddings = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
        self.vector_store = None
        self.qa_chain = None
        self.index = None  # FAISS index for similarity search
    
    def load_documents(self):
        """Load text documents from the specified directory."""
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir)
            print(f"Created documents directory: {self.documents_dir}")
            
        loader = DirectoryLoader(
            self.documents_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        print(f"Created {len(texts)} document chunks")
        return texts
    
    def create_vector_store(self, texts):
        """Create a FAISS vector store for fast similarity search."""
        try:
            self.vector_store = FAISS.from_documents(
                texts,
                self.embeddings
            )
            print("Vector store created successfully")
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
    
    def setup_qa_chain(self):
        """Setup the retrieval-augmented generation (RAG) system with OpenAI model."""
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            api_key=st.secrets['OPENAI_API_KEY']
        )
        
        prompt_template = """
        You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Try to provide specific examples and details from the context when possible.
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that.
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={
                "prompt": PROMPT
            }
        )
        print("QA chain setup completed")
    
    def initialize_system(self):
        """Initialize the RAG system, load documents, create vector store, and setup QA chain."""
        print("\nInitializing RAG system...")
        print("Step 1: Loading documents...")
        documents = self.load_documents()
        
        print("\nStep 2: Processing documents...")
        texts = self.split_documents(documents)
        
        print("\nStep 3: Creating vector store...")
        self.create_vector_store(texts)
        
        print("\nStep 4: Setting up QA chain...")
        self.setup_qa_chain()
        
        print("\nRAG system initialized successfully!")
    
    def query(self, question):
        """Query the RAG system with a question and return an answer."""
        if not self.qa_chain:
            raise Exception("System not initialized. Please run initialize_system() first.")
        
        try:
            response = self.qa_chain.invoke({"query": question})
            return response["result"]
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def save_vector_store(self, path="vector_store"):
        """Save the vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"Vector store saved to {path}")
        else:
            print("No vector store to save")
    
    def load_vector_store(self, path="vector_store"):
        """Load the vector store from disk."""
        if os.path.exists(path):
            try:
                self.vector_store = FAISS.load_local(path, self.embeddings)
                print(f"Vector store loaded from {path}")
                self.setup_qa_chain()
            except Exception as e:
                print(f"Error loading vector store: {str(e)}")
        else:
            print(f"No vector store found at {path}")
    
    def initialize_faiss_index(self):
        """Initialize the FAISS index if not already loaded."""
        vector_store_path = 'vector_store/index.faiss'

        if not os.path.exists(vector_store_path):
            self.index = faiss.IndexFlatL2(self.embeddings.embedding_size)  # L2 distance index
            print("FAISS index created.")
            # Save an empty index
            faiss.write_index(self.index, vector_store_path)
        else:
            try:
                self.index = faiss.read_index(vector_store_path)
                print("FAISS index loaded.")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")

