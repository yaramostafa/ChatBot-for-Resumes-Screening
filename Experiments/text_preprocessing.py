from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load and split the PDF document and return the documents and text chunks
def load_split_pdf(file_path):
    loader = PyPDFLoader(file_path)  
    documents = loader.load() 
    
    min_chunk_size = 100
    max_chunk_size = 700
    chunk_overlap = 50
    
    # recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )   

    # Split the loaded documents into chunks
    chunks = text_splitter.split_documents(documents)
    return documents, chunks

path = 'CVs'
for doc_name in os.listdir(path):
    # Get the full file path
    doc_path = os.path.join(path, doc_name)

    # Check if the file is a PDF or Word document
    if os.path.isfile(doc_path) and (doc_path.endswith('.pdf') or doc_path.endswith('.docx')):
        # Now load and split the document
        documents, chunks = load_split_pdf(doc_path) if doc_path.endswith('.pdf') else load_split_pdf(doc_path)

        # Print results
        print("docs",len(documents))
        print("chunks",chunks)
        print("++++++++++++++++++++++")
