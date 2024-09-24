# from https://python.langchain.com/docs/tutorials/agents/

# mycolab: https://colab.research.google.com/drive/14FVfOAfUnOXI9dxUT11hlIUJA5y0hMNS#scrollTo=WV0qhtc7m6iX

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

import datetime

import time

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

class StreamlitCallbackHandler:
    def __init__(self, container):
        self.container = container

    def __call__(self, chunk):
        self.container.write(chunk["output"])

def agents_app():
    # Streamlit app
    st.subheader('LangChain Agents')

    url = "https://python.langchain.com/docs/tutorials/agents/"

    st.markdown("The engine of the app is here: [Build an Agent](%s)" % url)

    search_query = st.text_input("Search Query")
    chunks = ""

   
    if st.button("Search"):
        try:
            with st.spinner("Please wait ..."):
                time.sleep(1)  # wait for 3 seconds
                chunks = agents_sl(search_query=search_query)
                print("this is in streamlit")
                print(chunks)
                #result_area('\n'.join(map(str, chunks)))
                if chunks:
                    result_area = st.text_area("Results", value=chunks)
                time.sleep(1)
            st.success(f"Ou yeah . . . {search_query}")
        except Exception as e:
            st.exception(f"An error occurred: {e}")

       

def agents():
    print("main")
    # Create the agent
    memory = MemorySaver()
    model = ChatGroq()
    search = TavilySearchResults(max_results=2, include_answer=True)
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}


    # Use the agent - test
    '''    
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="hi im Dare! and i live in SF")]}, config
    ):
        print(chunk)
        print("----")

    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
    ):
        print(chunk)
        print("----")
    '''
    # Use the agent  - in terminal
    
    while True:
        user_input = input("Enter a message (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit']:
            break

        config = {"configurable": {"thread_id": "abc123"}}
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}, config):
            print(chunk)
            print("----")

# Use the agent  - in streamlit app
def agents_sl(search_query):
    # Create the agent
    memory = MemorySaver()
    model = ChatGroq()
    search = TavilySearchResults(max_results=2, include_answer=True)
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}

        # Use the agent - test
    chunks = []
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=search_query)]}, config
    ):  
        now = datetime.datetime.now()

        chunks.append(chunk)

        print(f" now: {now} and chunk is: {chunk}")
        print("----")
        #pass
    return chunks









if __name__== "__main__":
    agents()
'''
if __name__ == "__main__":
    search_query = st.text_input("Enter your search query:", key="search_query")
    if search_query:
        agents_sl(search_query)
'''