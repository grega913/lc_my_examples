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
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

import time

from . import helperz

import streamlit as st

from .helperz import chain_basic, chain_basic_sl, chain_basic_history_sl






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


def chain_with_history_app(session_id:str):
    # Streamlit app
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
                time.sleep(2)

                conversational_rag_chain = chain_basic_history_sl(input=input_text)
                answer = conversational_rag_chain.invoke({
                    "input": input_text},
                    config={
                        "configurable": {"session_id": session_id}
                    },  # constructs a key "abc123" in `store`.
                )["answer"]
                st.write("Answer:", answer)

                session_history = conversational_rag_chain.get_session_history(session_id=session_id)
                
                st.write("Session History:", session_history)


                time.sleep(1)
            st.success(f"Ou yeah . . {answer}")
        except Exception as e:
            st.exception(f"An error occurred: {e}")


def agents_app():
    # Streamlit app
    st.title("Agents")
    st.markdown("This example with the use of Agents.")
    '''
    search_query = st.text_input(
        "Search Query - ask a question regarding the paper?",
         placeholder="What is Task Decomposition?",
         )
  
    if st.button("Search", key="search_btn", on_click=lambda: None):
        try:
            with st.spinner("Please wait ..."):
                search_query = st.session_state.search_query
                chain_result = chain_basic_sl(search_query)
                result_area = st.text_area("Results", value=chain_result)
                time.sleep(1)
            st.success(f"Ou yeah . . . {search_query}")
        except Exception as e:
            st.exception(f"An error occurred: {e}")
    '''


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