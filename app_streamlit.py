
import streamlit as st

from examples.search import search_app
from examples.playground import intro, mapping_demo, plotting_demo, data_frame_demo
from tutorials.agents import agents_app
from tutorials.qa_chat_history import qa_chat_history_app
import os
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="LangChain examples by GS", page_icon="🌶️")



page_names_to_funcs = {
    "—": intro,
    "Plotting Demo": plotting_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo,
    "SearchApp" : search_app,
    "AgentApp" : agents_app,
    "QaChatHistory": qa_chat_history_app
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
