## 🏗 Our System Architecture

![image](https://github.com/user-attachments/assets/605ed59f-1fbc-49bf-9a83-dbd46b2e938a)

![image](https://github.com/user-attachments/assets/e4ad7af2-e064-4f19-8923-d99344e3b63b)

## ✨ Features
- PDF and DOCX CV processing
- Intelligent text chunking and embedding
- Vector database storage using Pinecone
- RAG-based query processing
- Interactive chat interface
  
## 📁 Project Structure
```
  cv-matching-assistant/
  ├── app.py                  # Main Streamlit application
  ├── files_reader_chunker.py # Document processing and chunker
  ├── cvs_processing.py       # Document processing module (Cv chunks uploader to DB)
  ├── vector_database.py      # Pinecone database operations
  ├── rag_pipeline.py         # RAG implementation
  ├── requirements.txt        # Project dependencies
  ├── .env                    # Environment variables
  └── Dockerfile
```       
  
## 🔧 Prerequisites
- Python 3.8+
- Pinecone API key
- Groq API key

## 📥 Installation
Clone the repository:
```
git clone https://github.com/yaramostafa/ChatBot-for-Resumes-Screening
cd cv-matching-assistant
```
Build Docker:
```
docker build -t cv-matching-chatbot .
```
Run on:
```
docker run -p 8501:8501 cv-matching-chatbot
```

Access on: http://localhost:8501

## 💻 Our UI

![image](https://github.com/user-attachments/assets/e8f2295a-148f-43a6-8e7c-83cbb7b8632c)



