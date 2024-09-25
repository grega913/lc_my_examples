import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, HumanMessage


import os
from dotenv import load_dotenv
load_dotenv()


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

# function modified to work with streamlit
def chain_basic_sl(input:str):
    print("def chain_basic_sl")
    
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

    response = rag_chain.invoke({"input": input})

    return response["answer"]



store = {}

def chain_basic_history_sl():
    print("chain_basic_history_sl")

    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOpenAI()


    ### Construct retriever ###
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()


    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
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
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### Statefully manage chat history ###
    #store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # session_history = get_session_history(session_id=session_id)

    # print("session_history: ", str(session_history))
    return conversational_rag_chain
    '''
    answer = conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
        )["answer"]
    

    print("input: ", input)
    print("answer: ", answer)
    
    session_history = get_session_history(session_id="abc123")
    print("session_history: ", session_history)
    '''




# Agents #
# https://python.langchain.com/docs/tutorials/qa_chat_history/#agents
def agents_sl():
    print("def agents_sl")

    memory = MemorySaver()
    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm= ChatOpenAI()
    loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    ### Build retriever tool ###
    tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
    )
    tools = [tool]

    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    return agent_executor







if __name__ == "__main__":

    '''
    conversational_rag_chain = chain_basic_history_sl()

    input = "What is Task Decomposition?"
    answer = conversational_rag_chain.invoke({
        "input": input},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"]

    print("input: ", input)
    print("answer: ", answer)
    '''

    # test Agent
    agent_executor = agents_sl()
    config = {"configurable": {"thread_id": "abc123"}}
    query = "What is Task Decomposition?"
    for s in agent_executor.stream({"messages": [HumanMessage(content=query)]}, config=config):
        print(s)
        print("----")




