from files_reader_chunker import DocumentProcessor
from vector_database import PineconeDB

input_folder = "CVs"
processor = DocumentProcessor()
chunks = processor.process_folder(input_folder)

vector_databases = PineconeDB()
vector_databases.upload_chunks_to_pinecone(chunks)