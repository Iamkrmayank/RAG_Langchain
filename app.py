import streamlit as st
import os
from rag_system import RAGSystem

def initialize_rag_system():
    if 'rag_system' not in st.session_state:
        rag = RAGSystem()
        try:
            # Try to load existing vector store
            if os.path.exists('vector_store'):
                rag.load_vector_store('vector_store')
            else:
                rag.initialize_system()
                rag.save_vector_store('vector_store')
            st.session_state['rag_system'] = rag
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
            return False
    return True

def main():
    st.title("📚 RAG Query System")
    st.write("Ask questions about your documents using AI!")

    # Initialize RAG system
    if not initialize_rag_system():
        st.stop()

    # Document upload section
    st.sidebar.header("📄 Document Management")
    uploaded_file = st.sidebar.file_uploader("Upload new document", type=['txt'])
    
    if uploaded_file:
        try:
            # Ensure the documents directory exists
            os.makedirs("documents", exist_ok=True)

            # Save uploaded file
            save_path = os.path.join("documents", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")

            # Reinitialize the system with new document
            st.session_state['rag_system'].initialize_system()
            st.session_state['rag_system'].save_vector_store()
            st.sidebar.success("System reinitialized with new document!")
        except Exception as e:
            st.sidebar.error(f"Error processing upload: {str(e)}")

    # Query section
    st.header("🔍 Ask a Question")
    
    # Get user question
    question = st.text_area("Enter your question:")
    
    if st.button("Get Answer"):
        if question:
            try:
                with st.spinner("Thinking..."):
                    answer = st.session_state['rag_system'].query(question)
                st.write("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
        else:
            st.warning("Please enter a question.")

    # Display current documents
    st.sidebar.header("📚 Current Documents")
    if os.path.exists("documents"):
        docs = os.listdir("documents")
        if docs:
            for doc in docs:
                st.sidebar.text(f"• {doc}")
        else:
            st.sidebar.text("No documents uploaded yet.")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    This RAG (Retrieval Augmented Generation) system allows you to:
    - Upload text documents
    - Ask questions about your documents
    - Get AI-powered answers based on your content
    """)

if __name__ == "__main__":
    main()
