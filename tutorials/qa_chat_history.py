# from https://python.langchain.com/docs/tutorials/qa_chat_history/
# mycolab: https://colab.research.google.com/drive/1TkeZ2ZriVQUPlkS1UUMfVYryF1y-ZFYf#scrollTo=ZYa36_a07OQc


import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver


'''import os
from dotenv import load_dotenv
load_dotenv()
'''

import time

from . import helperz

import streamlit as st

from .helperz import chain_basic, chain_basic_sl, chain_basic_history_sl, agents_sl





# streamlit part for basic chain
def chain_basic_app():
    # Streamlit app
    st.title("ChainBasic")

    st.markdown("This is the simplest example. With just a simple chain.")

    search_query = st.text_input(
        "Search Query - ask a question regarding the paper?",
         placeholder="What is Task Decomposition?",
         )
  
    if st.button("Search", key="btn1"):
        try:
            with st.spinner("Please wait ..."):
   
                chain_result = chain_basic_sl(search_query)
                result_area = st.text_area("Results", value=chain_result)
                time.sleep(1)
            st.success(f"Ou yeah . . . {search_query}")
        except Exception as e:
            st.exception(f"An error occurred: {e}")


# streamlit part for chain with history
def chain_with_history_app(session_id:str):

    conversational_rag_chain = None
    if conversational_rag_chain== None:
        conversational_rag_chain = chain_basic_history_sl()

    st.title("ChainWithHistory")
    st.markdown("This example is with history.")


    input_text = st.text_input(
        "Search Query - ask a question regarding the paper?",
         placeholder="What is Task Decomposition?",
         key="it1"
    )
  
    if st.button("Search", key="btn2"):
        try:
            with st.spinner("Please wait ..."):
                        
                
                answer = conversational_rag_chain.invoke({
                    "input": input_text},
                    config={
                        "configurable": {"session_id": session_id}
                    },
                )["answer"]

                st.write("Input:", input_text)

                st.write("Answer:", answer)

                session_history = conversational_rag_chain.get_session_history(session_id=session_id)
                
                st.write("Session history: ", session_history)
                


                
            st.success(f"Ou yeah . . done for now, but please keep talking...")
        except Exception as e:
            st.exception(f"An error occurred: {e}")


# streamlit part for agents
def agents_app():

    agent_executor=None
    config = None

    if agent_executor==None:
    # we are initializing agent here, as we don't want to call init with every click
        agent_executor = agents_sl() 
    
    if config==None:
        config = {"configurable": {"thread_id": "abc123"}}

    # Streamlit app
    st.title("Agents")
    st.markdown("This example with the use of Agents.")

    input_text = st.text_input(
    "Search Query - ask a question regarding the paper?",
        placeholder="What is Task Decomposition?",
        key="it3"
    )


    if st.button("Search", key="btn3"):
        try:
            with st.spinner("Please wait ..."):
                time.sleep(2)

                for s in agent_executor.stream({"messages": [HumanMessage(content=input_text)]}, config=config):
                    print(s)
                    st.write(s)
                    st.write("----------")
                    print("----")
                
                time.sleep(2)

            st.success(f"Ou yeah . . done for now, but please keep talking...")
        except Exception as e:
            st.exception(f"An error occurred: {e}")


# streamlit part with 3 variants
def qa_chat_history_app():

    st.subheader('LangChain Qa_Chat_History')
    st.markdown("The engine of the app is here: [Conversational RAG](%s)" % "https://python.langchain.com/docs/tutorials/qa_chat_history/")
    st.markdown("The article is here: [LLM Powered Autonomous Agents](%s)" % "https://lilianweng.github.io/posts/2023-06-23-agent/")
    
    tab1, tab2, tab3 = st.tabs(["ChainBasic", "ChainWithHistory", "Agent"])

    
    with tab1:
        chain_basic_app()

    with tab2:
        chain_with_history_app(session_id="abc123")

    with tab3:
        agents_app()
    



    


if __name__== "__main__":
    answer = chain_basic()
    print(answer)