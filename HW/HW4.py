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
            candidate = section + i + 1
            offset = abs(candidate - midpoint)
            if best_point is None or offset < best_point:
                best_point = offset
                split_here = candidate
    chunk_1 = text[:split_here].strip()
    chunk_2 = text[:split_here].strip()
    return chunk_1, chunk_2

    

def add_chunks_to_collection(collection, chunk_1, chunk_2, file_name):
    #Creating an embedding from pdf
    client = st.session_state.client
    for i, chunk in enumerate([chunk_1, chunk_2], start=1)
    response = client.embeddings.create(
        input= chunk,
        model= 'text-embedding-3-small'
    )
    #Get the embedding
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
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab4')
    collection = chroma_client.get_or_create_collection('Lab4Collection')
    load_htmls_to_collection('./html_files_hw4', collection)
    return collection

if 'HW4_VectorDB' not in st.session_state:
    st.session_state.Lab4_VectorDB = create_hw4_vectordb()

collection = st.session_state.Lab4_VectorDB
st.title("HW4 RAG chatbot")
st.write("Chatbot Demo")

    
model = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

buffer_type = st.sidebar.selectbox('Buffer type', ('Last 2 responses', 'Token-based'))

base_system_prompt = ("Explain all answers simply enough for a 10-year-old to understand.After the user's first response ask them this:Do you want more information?. "
            "If they say yes, give them more information and then ask them specifically: Do you want more information?. If they say no, ask them specifically How can I help you?")





def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def msg_buffer(messages, system_prompt):
    return [system_prompt] + messages[-4:]

max_tokens = 5000

#Note to Grader:  I used AI to strategize how to calculate the tokens and for the coding logic 
# on looking at the last messages
def token_buffer(messages, system_prompt, max_tokens, model=model):
    system_tokens = count_tokens(system_prompt["content"], model)
    budget = max_tokens - system_tokens
    kept = []
    total = 0
    for msg in reversed(messages):
        t = count_tokens(msg["content"], model)
        if total + t > budget:
            break
        kept.insert(0, msg)
        total += t
    return [system_prompt] + kept   

if prompt := st.chat_input("What is up?"):    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- RAG retrieval: fetch relevant course material for this question ---
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
        + "\n\nYou have the following course material retrieved for this question. "
          "If you use it in your answer, clearly say so (e.g., 'Based on the course materials...'). "
          "If it isn't relevant, answer from general knowledge and say you are not using the course materials.\n\n"
        + context_text
    }

    if buffer_type == "Last 2 responses":
        api_msg = msg_buffer(st.session_state.messages, system_prompt)
    else:
        api_msg = token_buffer(st.session_state.messages, system_prompt, max_tokens, model)

    stream = client.chat.completions.create(
        model= model,
        messages = api_msg,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})