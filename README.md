## 20240925 - Qa_Chat_History

- https://python.langchain.com/docs/tutorials/qa_chat_history/
- qa_chat_history_app - main function
- 3 ways described:
  ** simple chain
  ** chain with memory
  \*\* agents ()
  Make sure to initialize chain or agent only once and then invoking or streaming with inputs from text_input multiple times, in order for memory to work ok.

## Issues

1. Loading tabs - problem is that all functions get loaded within tab blocks at init, and we need to change that. Conditional rendering is currently not supported.
   https://docs.streamlit.io/develop/api-reference/layout/st.tabs

## What's here?

This is a streamlit app with various pieces from LangChain tutorials and HowTos.

- Tutorials: https://python.langchain.com/docs/tutorials/
- HowTos: https://python.langchain.com/docs/how_to/

Most of the code is just rewritten from personal Colab Notebooks and modified/adapted for Streamlit.

## Usage

1.  git clone
2.  create a virtual environment
3.  pip install -r requirements.txt
4.  API KEYS are needed for communicating with Providers. You should store them in your personal .env file
    OPENAI_API_KEY=""
    SERPER_API_KEY=""
    GROQ_API_KEY=""
    TAVILY_API_KEY=""

## Run

1.  streamlit run app_streamlit.py
