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
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st





# https://colab.research.google.com/drive/1TkeZ2ZriVQUPlkS1UUMfVYryF1y-ZFYf#scrollTo=YRxmzEJf8YLb&line=39&uniqifier=1
def chain_basic():

    llm = ChatOpenAI(model="gpt-4o-mini") 
   # 1. Load, chunk and index the contents of the blog to create a retriever.

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )


    # Example of WebBaseLoader loading xml file
    '''
    loader = WebBaseLoader(
        "https://www.govinfo.gov/content/pkg/CFR-2018-title10-vol3/xml/CFR-2018-title10-vol3-sec431-86.xml"
    )
    loader.default_parser = "xml"
    '''


    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()


    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": "What is Task Decomposition?"})

    return response["answer"]


if __name__== "__main__":
    answer = chain_basic()
    print(answer)