# Import necessary libraries
import os
import tempfile
import streamlit as st
import embedchain
from embedchain import App
import base64
from streamlit_chat import message

# Configure the Embedchain Bot
def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.2:latest",
                    "max_tokens": 250,
                    "temperature": 0.5,
                    "stream": True,
                    "base_url": "http://localhost:11434"
                }
            },
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.2:latest",
                    "base_url": "http://localhost:11434"
                }
            },
        }
    )

# Function to display the PDF in Streamlit
def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Set up the Streamlit App
st.title("Chat with PDF using Llama 3.2")
st.caption("This app allows you to chat with a PDF using Llama 3.2 running locally with Ollama!")

# Set up temporary database path
db_path = tempfile.mkdtemp()

# Initialize session state
if "app" not in st.session_state:
    st.session_state.app = embedchain_bot(db_path)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload and preview
with st.sidebar:
    st.header("PDF Upload")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)

# Add PDF to the knowledge base
if pdf_file and st.button("Add PDF to Knowledge Base"):
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            st.session_state.app.add(f.name, data_type="pdf_file")
        os.remove(f.name)
    st.success(f"Added {pdf_file.name} to the knowledge base!")

# Chat interface
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=(msg["role"] == "user"), key=str(i))

# Define the fixed prompt
fixed_prompt = """For the user question <question> {question} </question>, 
    Read the following context \n <document> {context} </document>, \n 
    answer the user question from the given context, the answer must be clear and precise based on the context only."""


if prompt := st.chat_input("Ask a question about the PDF"):
    # Append user query to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    # Extract relevant context
    with st.spinner("Extracting context..."):
        #context = st.session_state.app.extract_context(prompt, data_type="pdf_file")
        context = st.session_state.app.query(prompt)
        if not context:
            context = "No relevant context found in the document."

    # Format the full prompt
    full_prompt = fixed_prompt.replace("{question}", prompt).replace("{context}", context)

    # Get response from the bot
    with st.spinner("Thinking..."):
        response = st.session_state.app.chat(full_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response)

# Button to clear chat history
if st.button("Clear History"):
    st.session_state.messages = []
    st.experimental_rerun()
