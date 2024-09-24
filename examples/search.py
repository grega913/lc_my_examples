# https://github.com/alphasecio/langchain-examples/tree/main/search

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools

import os
from dotenv import load_dotenv
load_dotenv()



env_openai_api_key = os.environ.get("OPENAI_API_KEY")
env_serper_api_key = os.environ.get("SERPER_API_KEY")


# print(env_openai_api_key, env_serper_api_key)




def search_app():

    openai_api_key_val = ""
    serper_api_key_val = ""

    # Streamlit app
    st.subheader('LangChain Search')

    # Get OpenAI API key, SERP API key and search query
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API key", value="", type="password")
        st.caption("*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
        serpapi_api_key = st.text_input("SERP API Key", type="password")
        st.caption("*If you don't have a SERP API key, get it [here](https://serpapi.com).*")
    search_query = st.text_input("Search Query")

    # If the 'Search' button is clicked
    if st.button("Search"):

        if openai_api_key:
            openai_api_key_val = openai_api_key
        else:
            openai_api_key_val = env_openai_api_key

        if serpapi_api_key:
            serper_api_key_val = serpapi_api_key
        else:
            serper_api_key_val = env_serper_api_key 

        print("openai_api_key_val: ", openai_api_key_val)
        print("serper_api_key_val: ", serper_api_key_val)


        # Validate inputs
        if not openai_api_key_val or not serper_api_key_val or not search_query.strip():
            st.error(f"Please provide the missing fields.")
        else:
            try:
                with st.spinner('Please wait...'):
                # Initialize the OpenAI module, load the SerpApi tool, and run the search query using an agent
                    

                    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key_val, verbose=True)

                    tools = load_tools(["serpapi"], llm, serpapi_api_key=serper_api_key_val)
                    
                    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
                    
                    result = agent.run(search_query)
                    st.success(result)
            except Exception as e:
                st.exception(f"An error occurred: {e}")

    
    st.code("""
        def search_app():
            # Get OpenAI API key, SERP API key and search query
                with st.sidebar:
                    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
                    st.caption("*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
                    serpapi_api_key = st.text_input("SERP API Key", type="password")
                    st.caption("*If you don't have a SERP API key, get it [here](https://serpapi.com).*")
                search_query = st.text_input("Search Query")

                # If the 'Search' button is clicked
                if st.button("Search"):
                    # Validate inputs
                    if not openai_api_key.strip() or not serpapi_api_key.strip() or not search_query.strip():
                        st.error(f"Please provide the missing fields.")
                    else:
                        try:
                            with st.spinner('Please wait...'):
                            # Initialize the OpenAI module, load the SerpApi tool, and run the search query using an agent
                                llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, verbose=True)
                                tools = load_tools(["serpapi"], llm, serpapi_api_key=serpapi_api_key)
                                agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
                                result = agent.run(search_query)
                                st.success(result)
                        except Exception as e:
                            st.exception(f"An error occurred: {e}")
        """)



if "__name__" == "__main__":
    search_app()