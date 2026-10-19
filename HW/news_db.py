import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pandas as pd
import chromadb
import streamlit as st
from openai import OpenAI
 
client = OpenAI(api_key=st.secrets["OPENAI_SECRET_KEY"])
 
df = pd.read_csv("hw7_files/news.csv").dropna(subset=["Document"])
 
chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_HW7")
collection = chroma_client.get_or_create_collection("HW7Collection")
 
if collection.count() == 0:
        docs = df["Document"].tolist()
 
        response = client.embeddings.create(input=docs, model="text-embedding-3-small")
 
        collection.add(
            ids=[str(i) for i in df.index],
            documents=docs,
            embeddings=[e.embedding for e in response.data],
            metadatas=[{"company": r.company_name, "date": r.Date, "url": r.URL}
                       for r in df.itertuples()]
        )
 
st.write("Articles in DB:", collection.count())