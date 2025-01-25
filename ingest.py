from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader, DirectoryLoader
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
# Newly added
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH="/home/ubuntu/BITS/SEM_3/ConvAI/llm_and_rag/diff_model/data"
DB_FAISS_PATH="/home/ubuntu/BITS/SEM_3/ConvAI/llm_and_rag/diff_model/vectorstore/db_faiss"

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    #model_kwargs = {'device': 'cpu'})
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()