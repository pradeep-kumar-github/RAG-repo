'''
1. Read in Knowledge Base
2. Turn documents into chunks
3. Vectorize the chunks
4. Store in Chroma
'''

import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

MODEL = "llama3"

DB_NAME = str(Path(__file__).parent.parent / "vector_db2")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# load_dotenv(override=True)

# frontier model = text-embedding-3-large
#embeddings = OpenAIEmbeddings(model = "MODEL_NAME")
embeddings = HuggingFaceEmbeddings(model = "BAAI/bge-base-en-v1.5")


def fetch_documents():
    documents = []
    
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))

    for folder in folders:
        doc_type = os.path.basename(folder)  #products, contracts, company, employees
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}) #load directory into memory
        folder_docs = loader.load() # fetches document's metadata
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    
    return documents
        

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embiddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embiddings(chunks)
    print("Ingestion complete!")