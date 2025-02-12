from files_reader_chunker import DocumentProcessor
import json
from datetime import datetime
from vector_database import PineconeDB

input_folder = "CVs"
processor = DocumentProcessor()
chunks = processor.process_folder(input_folder)

# vector_databases = PineconeDB()
# vector_databases.upload_chunks_to_pinecone(chunks)

# my chunks
if chunks:
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"cv_chunks_{timestamp}.json"
    
    # Save chunks to JSON file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"\nChunks saved to: {output_filename}")
        
        # Print some statistics
        print(f"Total chunks saved: {len(chunks)}")
        
    except Exception as e:
        print(f"Error saving chunks to JSON: {str(e)}")
else:
    print("No chunks were generated to save.")
    
# Print summary
print(f"\nProcessing complete!")
# Get unique file paths
unique_files = set(chunk['original_file'] for chunk in chunks)
print("\nUnique files processed:")
for file_path in unique_files:
    print(file_path)