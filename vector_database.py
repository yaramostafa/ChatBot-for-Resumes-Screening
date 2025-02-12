import os
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

class PineconeDB:
    def __init__(self, index_name="rag-cvs", embedding_dim=1024, region="us-east-1"):
        # Read API key from environment variable
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set.")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        
        # Create the index if it doesn't exist
        self.create_index()
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        
        # Initialize the embedding model
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    def create_index(self):
        """Creates the Pinecone index if it doesn't already exist."""
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Index '{self.index_name}' does not exist. Creating...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{self.index_name}' created.")
        else:
            print(f"Index '{self.index_name}' already exists.")
    
    def upload_chunks_to_pinecone(self, chunks):
        """Uploads chunks to Pinecone."""
        print(f"Starting upload of {len(chunks)} chunks to Pinecone DB.")
        
        for chunk in tqdm(chunks):
            # Use the chunk_id from the chunk itself
            chunk_id = chunk['chunk_id']
            
            # Get the embedding for the chunk content
            embedding = self.model.encode([chunk['content']])[0]
            
            # Prepare metadata
            metadata = {
                "original_file": chunk['original_file'],
                "chunk_id": chunk['chunk_id'],
                "content": chunk['content']
            }
            
            # Upload to Pinecone
            self.index.upsert(vectors=[(
                chunk_id,
                embedding.tolist(),  # Convert numpy array to list
                metadata
            )])
        
        print("Upload complete!")