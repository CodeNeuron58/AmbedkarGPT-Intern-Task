import os
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA

load_dotenv()

PERSIST_DIR = 'my_vectordb'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    """Returns the configured HuggingFace Embeddings instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_or_load_vector_store():
    """
    Checks if the vector store already exists on disk.
    If it exists, it loads it. Otherwise, it creates it from the text file and persists it.
    """
    embeddings = get_embeddings()
    
    if os.path.exists(PERSIST_DIR):
        print(f"Loading vector store from disk: {PERSIST_DIR}")
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        print("Vector store loaded successfully.")
        return vectordb
    else:
        print("Vector store not found. Building from source documents...")
        loader = TextLoader("speech.txt")
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )

        return vectordb

def get_llm():
    """Configures and returns the Cohere LLM instance."""
    api = os.getenv("COHERE_API_KEY")
    if not api:
        raise ValueError("COHERE_API_KEY environment variable not set.")
    llm = ChatCohere(model="command-a-03-2025", cohere_api_key=api)
    return llm

def main():
    vectordb = build_or_load_vector_store() 
    llm = get_llm()
    retriever = vectordb.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    print("\nEnter 'exit' to quit the application.")
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break
        else:
            result = qa_chain.invoke({"query": query})
            print("\nAnswer:")
            print(result["result"])
            print("-" * 20)

if __name__ == "__main__":
    main()
