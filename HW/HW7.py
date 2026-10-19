import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from openai import OpenAI
import tiktoken
import json
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import pandas as pd


if 'client' not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_SECRET_KEY"])
client = st.session_state.client

if 'HW7_VectorDB' not in st.session_state:
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW7')
    st.session_state.HW7_VectorDB = chroma_client.get_collection('HW7Collection')
collection = st.session_state.HW7_VectorDB


def search_news(query, n_results=10):
    query_embedding = client.embeddings.create(
        input=query, model='text-embedding-3-small'
    ).data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return [{"text": doc, **meta}
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
 
 
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search the news articles database and return the most similar articles",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic, company, or kind of news to search for"},
                    "n_results": {"type": "integer", "description": "How many articles to return (use about 20 for 'most interesting')"},
                },
                "required": ["query"],
            },
        },
    }
]
 
system_prompt = (
    "You are a news assistant for a large global law firm that monitors news about its clients. "
    "Only use articles returned by the search_news tool. Never make up articles. "
    "If asked for the most interesting news, search broadly, then return a numbered list ranked by "
    "importance to a law firm (lawsuits, regulation, investigations, mergers, and major financial events "
    "rank highest), with one sentence on why each one matters. "
    "If asked for news about a topic or company, search for it and summarize the matching articles. "
    "Include the company, date, and URL for each article."
)
 
 
def msg_buffer(messages):
    return [{"role": "system", "content": system_prompt}] + messages[-10:]
 
 
st.title("HW7 News Bot")
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-5.1"])
#extra = {} if model == "gpt-4o-mini" else {"reasoning_effort": "none"}


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me about the news."}]
 
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
 
if prompt := st.chat_input("Find the most interesting news"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
 
    messages = list(st.session_state.messages)
 
    response = client.chat.completions.create(
        model=model, 
        messages=msg_buffer(messages), 
        tools=tools, 
        tool_choice="auto"
    )
    response_message = response.choices[0].message
 
    if response_message.tool_calls:
        messages.append(response_message.to_dict())
        for tool_call in response_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = search_news(args["query"], args.get("n_results", 10))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })
        final = client.chat.completions.create(model=model, messages=msg_buffer(messages))
        answer = final.choices[0].message.content
    else:
        answer = response_message.content
 
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})