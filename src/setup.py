import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

#Directory to store vectorstore
persist_directory = "chroma_db"

pdf_path = "/home/ankursrivastava/Desktop/projects/assignment/data/document.pdf"  
loader = PyPDFLoader(pdf_path)
docs = loader.load()

#Split the chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)

#Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Create Chroma vectorstore
vectordb = Chroma.from_documents(
    chunks, 
    embeddings, 
    collection_name="documents", 
    persist_directory=persist_directory
)

#Persist to disk
vectordb.persist()
print(f"Vectorstore created with {len(chunks)} chunks and saved to {persist_directory}")