import os
from langchain_pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class RAG:
    def __init__(self, pc_index="rag-cvs", embed_model="BAAI/bge-large-en-v1.5",
                 llm_model='llama3-70b-8192'):
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbeddings(model_name=embed_model)

        # Initialize vector database
        self.vector_db = Pinecone.from_existing_index(
            index_name=pc_index,
            embedding=self.embed_model,
            text_key="content"
        )

        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=llm_model,
            temperature=0.0,
            streaming=True,
        )

        # Initialize improved memory with summary buffer
        self.mem_buff = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key='chat_history',
            return_messages=True,
            max_token_limit=2000,  # Adjust this based on your needs
            ai_prefix="Assistant",
            human_prefix="Human",
            summary_max_tokens=500  # Maximum length of summary
        )

        self.qa_creation()

    def qa_creation(self):
        """
        Function to create the QA chain with the prompts
        """
        system_template = """
        You are an HR assistant for CV matching. Use ONLY context information provided below.

        MAIN RULES:
            1. Use ONLY the provided context.
            2. If no relevant candidates match, respond with: 
            'No candidates found. Please refine the job description or required skills.'
            3. If a candidate is missing some requirements but is still relevant, mention the gaps in the response. 
            For example, 'This candidate does not meet all of the listed skills but is highly relevant based on their experience in X.'
            4. If multiple candidates match, rank them based on relevance, from the most relevant to the least relevant.
            5. DO NOT infer or create information beyond the given context. 

        FOR CANDIDATE RECOMMENDATIONS:
        1. Format response as:
        - Candidate Full Name
        - Skills
        - Analysis: Match reasoning

        FOR SPECIFIC CANDIDATE QUERIES:
        1. Include:
        - Candidate name
        - Requested details (experience/skills/roles)
        - Only provided information

        IF NO MATCHES:
        - Reply: "No candidates found with these skills. Please provide more skills or a better description."
        - OR ask clarifying questions

        Context:
        {context}
        """
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]

        chat_prompt = ChatPromptTemplate.from_messages(messages)

        # Initialize QA chain with configured retriever
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 50,
                    "lambda_mult": 0.5
                    }  
            ),
            memory=self.mem_buff,
            combine_docs_chain_kwargs={"prompt": chat_prompt},
            verbose=True
        )

    def check_db_content(self, query):
        """
        Function to check contents of vector database for a given query
        """
        print(f"\nChecking database content for query: '{query}'")
        docs = self.vector_db.similarity_search(query, k=3)
        if not docs:
            print("No documents found in the database for this query.")
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(doc.page_content)
            print("-" * 50)
        return docs

    def get_response(self, text):
        """
        Function to get response from the QA chain with debug information
        """
        print("\nRetrieving relevant documents...")
        docs = self.check_db_content(text)
        
        if not docs:
            return "No candidates found with these skills. Please provide more skills or a better description."
        
        print("\nGenerating response...")
        response = self.qa({"question": text})
        return response["answer"]


# rag = RAG()

# # Create an interactive loop
# while True:
#     # Get user input
#     user_query = input("\nEnter your query (or 'quit' to exit): ")
    
#     # Check if user wants to quit
#     if user_query.lower() in ['quit', 'exit', 'q']:
#         print("Goodbye!")
#         break
    
#     # Get and print the response
#     response = rag.get_response(user_query)
#     print("\nResponse:")
#     print(response)