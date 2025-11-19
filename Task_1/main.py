from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()

def build_vector_store():
    loader = TextLoader("speech.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chromadb_store"
    )
    
    return vectordb

def get_vector_store():
    if os.path.exists("chromadb_store"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory="chromadb_store", embedding_function=embeddings)
        return vectordb
    else:       
        return build_vector_store()
    
def get_llm():
    api = os.getenv("COHERE_API_KEY")
    llm = ChatCohere(model="command-a-03-2025", cohere_api_key=api)
    return llm

def main():
    vectordb = get_vector_store()
    llm = get_llm()
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
    )
    while True:
        query = input("Enter your query: ")
        if query == "exit":
            break
        else:
            result = qa_chain.invoke(query)
            print(result["result"])

if __name__ == "__main__":
    main()