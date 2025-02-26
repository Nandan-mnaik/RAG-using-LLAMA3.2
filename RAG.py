import os
import faiss
import numpy as np
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# FAISS Index setup
class FAISSIndex:
    def __init__(self, embedding_dim, index_file=None):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = {}  # Maps FAISS indices to document chunks
        self.index_file = index_file
        
        if index_file and os.path.exists(index_file):
            print("Loading existing FAISS index.")
            self.index = faiss.read_index(index_file)
        else:
            print("Creating a new FAISS index.")

    def add(self, embeddings, metadata):
        indices = list(range(len(self.metadata), len(self.metadata) + len(embeddings)))
        self.index.add(np.array(embeddings).astype('float32'))
        for i, meta in zip(indices, metadata):
            self.metadata[i] = meta

    def save(self):
        if self.index_file:
            faiss.write_index(self.index, self.index_file)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array(query_embedding).reshape(1, -1), k)
        results = [self.metadata[i] for i in indices[0] if i != -1]
        return results

# Step 1: Load the PDF document
loader = UnstructuredPDFLoader(file_path="MotorACT.pdf")
data = loader.load()

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Step 3: Initialize FAISS index
embedding_dim = 384  # For LLaMA embeddings
embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

faiss_index_file = "faiss_index_MotorACT.idx"
faiss_index = FAISSIndex(embedding_dim, index_file=faiss_index_file)

# Add embeddings to the FAISS index
if os.path.exists(faiss_index_file):
    print("Loaded existing FAISS index.")
else:
    embeddings = []
    metadata = []
    for chunk in chunks:
        try:
            embedding = embedding_function.embed(chunk.page_content)
            embeddings.append(embedding)
            metadata.append({"text": chunk.page_content})
        except Exception as e:
            print(f"Error generating embedding: {e}")
    faiss_index.add(embeddings, metadata)
    faiss_index.save()

# Step 4: Define the query and retrieval mechanism
class FAISSRetriever:
    def __init__(self, index):
        self.index = index

    def retrieve(self, query, k=5):
        embedding = embedding_function.embed(query)
        return self.index.search(embedding, k)

retriever = FAISSRetriever(faiss_index)

# Step 5: Define the LLM prompt and Chat model
llm = ChatOllama(model="llama3", show_progress=True)

template = """Read the documents. Understand the query. Identify relevant passages. Use passage information to provide a comprehensive, informative, clear, concise, and direct answer to the query. Combine information from multiple passages if necessary.
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Step 6: Define the RAG pipeline
def get_answer(question):
    if not question:
        return "Please enter a question."

    try:
        # Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(question, k=5)
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

        # Generate the answer using the context
        chain = (
            {"context": context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    user_question = input("Enter your legal question: ")
    response = get_answer(user_question)
    print("\nAnswer:\n", response)
