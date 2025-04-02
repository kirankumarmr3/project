import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
#from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] =  os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
st.title("RAG Document Q&A With Groq And Lama3")
user_input=st.text_input("Enter your query from the research paper")

if st.button("prepare_Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
    
llm = ChatGroq(model_name="deepseek-r1-distill-qwen-32b", groq_api_key=groq_api_key)

# context is used when we are sending the external document along with prompt,   only {context} is enough, no need to give  --> <context> {context} <context>,
# and when we are passing user input then it should be {input}.
prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the given context only.
    <context> {context} <context>
     Question:{input}
    """
)  

#user_prompt=st.text_input("Enter your query from the research paper")
if user_input:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()  # as_retriever acts as a interface to pass the query to the vector store
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    #print(retrieval_chain.input_keys) # added for testing
    response=retrieval_chain.invoke({'input':user_input})
    # print(response["answer"])
    st.write(response['answer'])
    
    
    
    
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )