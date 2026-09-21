import sys


__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from openai import OpenAI
import tiktoken
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup


if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_SECRET_KEY"]
    st.session_state.client= OpenAI(api_key=api_key)

def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text

def chunk_text_into_two(text):
    midpoint = len(text) //2

    window = 200
    start = max(0, midpoint - window)
    end = min(len(text), midpoint + window)
    section = text[start:end]
    best_point = None
    split_here = midpoint

    for i, ch in enumerate(section):
        if ch == "." and i + 1 < len(section) and section[i + 1] == " ":
            candidate = start + i + 1
            offset = abs(candidate - midpoint)
            if best_point is None or offset < best_point:
                best_point = offset
                split_here = candidate
    chunk_1 = text[:split_here].strip()
    chunk_2 = text[split_here:].strip()
    return chunk_1, chunk_2

    

def add_chunks_to_collection(collection, chunk_1, chunk_2, file_name):
    #Creating an embedding from pdf
    client = st.session_state.client
    for i, chunk in enumerate([chunk_1, chunk_2], start=1):
        response = client.embeddings.create(
            input= chunk,
            model= 'text-embedding-3-small'
    )
        embedding = response.data[0].embedding

    #Add embedding and document to ChromaDB
        collection.add(
            documents=[chunk],
            ids=[f"{file_name}_chunk{i}"],
            metadatas=[{"filename": file_name, "chunk": i}],
            embeddings= [embedding]
        )                    
    return collection

def load_htmls_to_collection(folder_path, collection):
    if collection.count() == 0:
        html_dir = Path(folder_path)
        for html_file in html_dir.glob("*.html"):
            text = extract_text_from_html(html_file)
            chunk_1, chunk_2= chunk_text_into_two(text)
            add_chunks_to_collection(collection, chunk_1, chunk_2, html_file.name)

def create_hw4_vectordb():
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')
    collection = chroma_client.get_or_create_collection('HW4Collection')
    load_htmls_to_collection('./html_files_hw4', collection)
    return collection

if 'HW4_VectorDB' not in st.session_state:
    st.session_state.HW4_VectorDB = create_hw4_vectordb()

collection = st.session_state.HW4_VectorDB
st.title("HW4 RAG chatbot")
st.write("Chatbot Demo")

    
model = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

buffer_type = st.sidebar.selectbox('Buffer type', ('Last 2 responses', 'Token-based'))

base_system_prompt = ("Be a helpful assistant. Answer using the retrieved course material "
    "Use it when it is relevant and say: Based on the course "
    "If it isn't relevant, answer from general knowledge and say you are not using the course materials.")



def msg_buffer(messages, system_prompt):
    return [system_prompt] + messages[-10:]

if prompt := st.chat_input("What is up?"):    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    
    client = st.session_state.client
    query_response = client.embeddings.create(
        input=prompt,
        model='text-embedding-3-small'
    )
    query_embedding = query_response.data[0].embedding

    rag_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    retrieved_docs = rag_results['documents'][0]
    retrieved_ids = rag_results['ids'][0]

    context_text = "\n\n".join(
        f"[Source: {doc_id}]\n{doc_text[:1500]}" for doc_id, doc_text in zip(retrieved_ids, retrieved_docs)
    )

    system_prompt = {
        "role": "system",
        "content": base_system_prompt
        + "\n\netrieved course material:\n\n"
        + context_text
    }
    api_msg = msg_buffer(st.session_state.messages, system_prompt)

    stream = client.chat.completions.create(
        model= model,
        messages = api_msg,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})