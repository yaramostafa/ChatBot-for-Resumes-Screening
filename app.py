import streamlit as st
from files_reader_chunker import DocumentProcessor
from vector_database import PineconeDB
from rag_pipeline import RAG
import os

# Configure Streamlit page
st.set_page_config(
    page_title="CV Matching Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAG()

def process_cvs(directory_path):
    """Process CVs from the specified directory"""
    try:
        processor = DocumentProcessor()
        chunks = processor.process_folder(directory_path)
        
        vector_db = PineconeDB()
        vector_db.upload_chunks_to_pinecone(chunks)
        
        st.sidebar.success(f"Successfully processed CVs from {directory_path}")
        return True
    except Exception as e:
        st.sidebar.error(f"Error processing CVs: {str(e)}")
        return False

# Sidebar
with st.sidebar:
    st.title("CV Processing")
    st.write("Upload and process CVs from a directory")
    
    # Directory input
    directory_path = st.text_input("Enter directory path containing CVs:", "CVs")
    
    # Process button
    if st.button("Process CVs"):
        if os.path.exists(directory_path):
            with st.spinner("Processing CVs..."):
                success = process_cvs(directory_path)
                if success:
                    st.session_state.rag_system = RAG()  # Reinitialize RAG system
        else:
            st.error("Directory not found!")

# Main chat interface
st.title("CV Matching Assistant")
st.write("Chat with the AI to find matching candidates for your job requirements.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What kind of candidate are you looking for?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_system.get_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add some helpful information at the bottom
with st.expander("💡 Tips for better results"):
    st.write("""
    - Be specific about required skills and experience
    - Include important qualifications or certifications
    - Mention preferred years of experience
    - Specify industry or domain expertise if relevant
    - Include any must-have technical skills
    """)