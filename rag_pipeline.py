import os
from langchain_pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class RAG:
    def __init__(self, pc_index="rag-cvs-cleaned", embed_model="BAAI/bge-large-en-v1.5",
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
        self.mem_buff = ConversationBufferWindowMemory(
            memory_key='chat_history',
            return_messages=True,
            k=6  # Adjust this to the number of turns you want to keep
        )


        self.llm_chain_creation()

    def llm_chain_creation(self):
        """
        Function to create the llm chain with the prompts
        """
        system_template = """
        You are an HR assistant specialized in job matching. 
        1- Given a job description with requirements(skills, experience.etc) you should recommend the candidates
        2- You should give scores to the candidate and tell what makes them better than the others.
        3- DON'T INFER ANSWERS, Use ONLY the provided context and prior conversation history.
        4- If asked follow-up questions about previously mentioned candidates, refer to the conversation history before retrieving new information.
        5- If asked for a full CV of a canidiate retrieve all the data of the candidate with details
        6- If asked for other candidates refer to the chat history to make sure not to recommend the same candidates twice
        7 - If no relevant candidates are found, respond with:
        'No candidates found with these skills. Please provide more skills or a better description.'
        NOTE : DO NOT MAKE UP OR INFER INFORMATION THAT IS NOT EXPLICITLY STATED IN THE DOCUMENTS, CONTEXT OR CHAT HISTORY.
        Context:
        {context}
        Your response should include:
        1- the score of the candidate
        2- Candidate full name
        3- their values 
        4- Analysis on why you see the candidate is fit for the position
        If multiple candidates are found, list them in order of relevance.
        """

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]

        chat_prompt = ChatPromptTemplate.from_messages(messages)

        self.llm_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 30,
                    "lambda_mult": 0.5
                }
            ),
            memory=self.mem_buff,
            combine_docs_chain_kwargs={"prompt": chat_prompt},
            verbose=True,
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
        response = self.llm_chain({"question": text})
        return response["answer"]