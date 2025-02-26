import faiss
import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
import json
import numpy as np
from PyPDF2 import PdfReader
import os

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.2:1b"  # Replace with your model name in Ollama

EMBEDDINGS_FILE = "embeddings.npy"
FAISS_INDEX_FILE = "faiss.index"

# --- PDF PROCESSING ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# --- FAISS DENSE RETRIEVAL ---
def create_dense_index(documents):
    """
    Creates or loads a FAISS index for dense retrieval using Sentence Transformers.
    """
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = embedding_model.encode(documents, show_progress_bar=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        # Save embeddings and FAISS index for future use
        np.save(EMBEDDINGS_FILE, embeddings)
        faiss.write_index(index, FAISS_INDEX_FILE)

    return index, embeddings

def retrieve_dense(query, index, embedding_model, documents, top_k=3):
    """
    Performs dense retrieval using FAISS and Sentence Transformers embeddings.
    """
    query_embedding = embedding_model.encode([query])[0]
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [documents[idx] for idx in indices[0]]

# --- SPARSE RETRIEVER (BM25) ---
def create_sparse_retriever(documents):
    """
    Creates a BM25 sparse retriever.
    """
    tokenized_corpus = [doc.split() for doc in documents]
    return BM25Okapi(tokenized_corpus)

def retrieve_sparse(query, bm25, documents, top_k=3):
    """
    Performs sparse retrieval using BM25.
    """
    tokenized_query = query.split()
    return bm25.get_top_n(tokenized_query, documents, n=top_k)

# --- OLLAMA INTEGRATION ---
def query_ollama_stream(prompt, model=OLLAMA_MODEL_NAME):
    """
    Queries the Ollama API and processes streaming JSON responses.
    """
    url = "http://localhost:11434/api/generate"  # Adjust API endpoint as needed
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 200,
    }
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        if response.status_code == 200:
            final_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        parsed_line = json.loads(line.decode("utf-8"))
                        if "response" in parsed_line:
                            final_response += parsed_line["response"]
                    except json.JSONDecodeError as e:
                        print("Error decoding JSON line:", line, e)
            return final_response.strip()
        else:
            print(f"Error querying Ollama: {response.status_code} {response.text}")
            return "Error: Could not generate response."
    except Exception as e:
        print(f"Error during API query: {e}")
        return "Error: Could not generate response."

# --- RAG PIPELINE ---
def rag_pipeline(query, documents):
    """
    Executes the RAG pipeline: retrieval and generation.
    """
    index, embeddings = create_dense_index(documents)
    bm25 = create_sparse_retriever(documents)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dense_results = retrieve_dense(query, index, embedding_model, documents, top_k=3)
    print(f"Dense results: {dense_results}")

    sparse_results = retrieve_sparse(query, bm25, documents)
    print(f"Sparse results: {sparse_results}")

    combined_results = dense_results + sparse_results
    print(f"Combined results: {combined_results}")

    context = " ".join(combined_results[:3])  # Use top 3 results as context
    prompt = f"Context: {context}\nQuery: {query}\nAnswer:"
    response = query_ollama_stream(prompt)
    return response

# --- MAIN ---
if __name__ == "__main__":
    # User-provided inputs
    pdf_path = input("Enter the path to the PDF file: ").strip()
    user_query = input("Enter your query: ").strip()

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found. Please provide a valid path.")
    else:
        # Extract text from the provided PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        # Split the extracted text into smaller, document-like chunks
        documents = [chunk.strip() for chunk in extracted_text.split("\n\n") if chunk.strip()]
        print(f"Extracted {len(documents)} documents from the PDF.")

        # Run the RAG pipeline
        response = rag_pipeline(user_query, documents)
        print("\nFinal Response:", response)
