import os, json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import streamlit as st


  # ⬅️ import function

# 🔑 Load .env file
load_dotenv()

# OPENAI_KEY = st.secrets["OPENAI_KEY"]
OPENAI_KEY = "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"

# 🔑 Fetch key from env
# OPENAI_KEY = os.getenv("OPENAI_KEY")

if not OPENAI_KEY:
    raise ValueError("OPENAI_KEY not found. Please check your .env file.")





def build_vector_db():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    docs = []

    for fname in os.listdir("data/menus"):
        if fname.endswith(".json"):
            with open(os.path.join("data/menus", fname)) as f:
                data = json.load(f)
                for item in data.get("menu", []):
                    content = f"{item['item']} - {item['description']} - ₹{item['price']}"
                    docs.append(Document(page_content=content))

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("vector_db")
    print("✅ Vector DB created!")

if __name__ == "__main__":
    build_vector_db()
