
# https://python.langchain.com/docs/tutorials/pdf_qa/

from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.base import Runnable


import os
from dotenv import load_dotenv
load_dotenv()


def loadDocs(fName: str):
    """
    Load a PDF file and return the text content as a LangChain document list.

    Args:
        file_name (str): The name of the PDF file to load.

    Returns:
        Document: A LangChain document containing the text content of the PDF file.
    """
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    file_path = os.path.join(script_dir, fName)


    loader = PyPDFLoader(file_path)

    docs = loader.load()
    
    print(len(docs))
    i=0
    for doc in docs:

        if i<10:
            print(i)
            print(doc.page_content)

    return docs


def createAChain(fName:str) -> Runnable: 

    llm = ChatGroq(model="llama3-8b-8192")

    docs = loadDocs(fName)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

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


    return rag_chain


if __name__== "__main__":
    #loadDocs(fName="nke-10k-2023.pdf")
    loadDocs(fName="2409.10482v2.pdf")