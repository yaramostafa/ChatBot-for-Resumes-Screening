# Use a base Python image
FROM python:3.10

# Set environment variables to avoid interactive prompts
ENV PIP_NO_CACHE_DIR=false \
    PYTHONUNBUFFERED=1 \
    PINECONE_API_KEY=${PINECONE_API_KEY} \
    GROQ_API_KEY=${GROQ_API_KEY}
 
 
# Set the working directory in the container
WORKDIR /app
 
# Copy the requirements file and install dependencies
COPY requirements.txt .
 
# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt
 
# Copy the rest of the application
COPY . .
 
# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]