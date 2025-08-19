import os
import pickle
import faiss
import numpy as np
import sqlite3
import streamlit as st
import requests
import time
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Access secrets directly using Streamlit's st.secrets
GEMINI_API_KEY = st.secrets["general"]["GEMINI_API_KEY"]
GEMINI_API_URL = st.secrets["general"]["GEMINI_API_URL"]


# Define file paths for FAISS index and metadata
BASE_DIR = os.getcwd()  # Get current working directory
INDEX_DIR = os.path.join(BASE_DIR, 'rag_data', 'faiss_index')
META_PATH = os.path.join(BASE_DIR, 'rag_data', 'metadata.pkl')

# Load FAISS index and metadata from disk
index_path = os.path.join(INDEX_DIR, 'faiss.index')
index = faiss.read_index(index_path)

with open(META_PATH, 'rb') as f:
    chunked_docs = pickle.load(f)

# Initialize SQLite database for storing chat history
DB_PATH = "chat_history.sqlite"

def setup_chat_db():
    """Create the SQLite database and the table for storing chat messages."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()

def record_chat(role: str, message: str):
    """Record a chat message into the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO chat_history (role, message) VALUES (?, ?)", (role, message))
        conn.commit()

def fetch_chat_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve the last `limit` chat messages from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT role, message, timestamp FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    return [{"role": r, "message": m, "timestamp": t} for r, m, t in reversed(rows)]

def clear_chat_history():
    """Remove all chat records from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_history")
        conn.commit()

# Set up the database
setup_chat_db()

# Initialize the embeddings model
embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Function to query the Gemini API and generate a response
def get_gemini_response(prompt: str, history: List[Dict[str, str]] = None, word_limit: int = 300) -> str:
    """Send a prompt to the Gemini API and return the response."""
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    content = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                content.append({"role": "user", "parts": [{"text": msg["message"]}]})
            elif msg["role"] == "assistant":
                content.append({"role": "model", "parts": [{"text": msg["message"]}]})
    
    content.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {"contents": content}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Check if the response status is OK

        data = response.json()
        if 'candidates' in data:
            response_text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Limit the response to the specified word count
            if word_limit and len(response_text.split()) > word_limit:
                response_text = ' '.join(response_text.split()[:word_limit]) + "..."
            
            return response_text
        else:
            st.error(f"Unexpected response format: {data}")
            return "Sorry, there was an error processing your request."
    
    except requests.exceptions.RequestException as e:
        st.error(f"Gemini API error: {e}")
        if response is not None:
            st.error(f"Response: {response.text}")
        return "Sorry, there was an error processing your request."

# Streamlit chatbot interface
st.title("RAG Chatbot")

# Initialize conversation history in the session if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "message": "Hello! How can I help you today?"}]

# Load previous chat history from the database into session state
chat_history = fetch_chat_history(limit=15)
for message in chat_history:
    st.session_state["messages"].append({"role": message['role'], "message": message['message']})

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Handle user input and response
user_input = st.chat_input("Type your message here:")

if user_input:
    # Save user input to the database and session state
    record_chat("user", user_input)
    st.session_state["messages"].append({"role": "user", "message": user_input})

    # Query the FAISS index for relevant context
    query_vec = embedding_model.embed_query(user_input)
    query_vec = np.array([query_vec], dtype='float32')
    faiss.normalize_L2(query_vec)

    if index.is_trained:
        scores, indices = index.search(query_vec, 5)  # Top 5 matches
    else:
        st.error("FAISS index is not trained.")
        indices = []

    # Build context from retrieved chunks
    context_text = "\n\n".join([f"[Source: {chunked_docs[idx]['source']}] {chunked_docs[idx]['text']}" for idx in indices[0]])

    # Construct the full prompt for Gemini
    prompt = f"Context:\n{context_text}\n\nQuestion: {user_input}"

    # Show loading spinner while waiting for Gemini's response
    with st.spinner("Processing..."):
        answer = get_gemini_response(prompt, st.session_state["messages"], word_limit=300)

    # Save assistant's response and update session state
    record_chat("assistant", answer)
    st.session_state["messages"].append({"role": "assistant", "message": answer})

    # Display user input and assistant's response
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in answer.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

# Button to clear chat history
if st.button("Clear Chat History"):
    clear_chat_history()
    st.session_state["messages"] = []
    st.success("Chat history cleared successfully.")


