import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from chromadb import Client
from chromadb.config import Settings

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        chromadb_dir: str = "chromadb_store",
        collection_name: str = "vector_db",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.chromadb_dir = chromadb_dir
        self.collection_name = collection_name
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )
        self.chroma_client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.chromadb_dir))

    def create_embeddings(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No documents were loaded from the PDF.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError("No text chunks were created from the documents.")
        
        collection = self.chroma_client.get_or_create_collection(self.collection_name)
        for idx, doc in enumerate(splits):
            embedding = self.embeddings.embed_query(doc.page_content)
            collection.add(
                ids=[str(idx)],
                documents=[doc.page_content],
                metadatas=[{"source": doc.metadata}],
                embeddings=[embedding],
            )
        
        self.chroma_client.persist()
        return "✅ Vector DB Successfully Created and Stored in ChromaDB!"
