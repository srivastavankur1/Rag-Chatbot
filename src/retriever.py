from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# def load_retriever():
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     retriever = Chroma(
#         collection_name="legal_docs",
#         embedding_function=embeddings,
#         persist_directory="chroma_db"
#     ).as_retriever()
#     return retriever


def load_retriever(persist_dir="chroma_db", collection="legal_docs", model="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model)

    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})