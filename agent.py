import re, json, os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st

from vector_index import build_vector_db   # ⬅️ import function

# 🔑 Load .env file
load_dotenv()


# OPENAI_KEY = st.secrets["OPENAI_KEY"]

# OPENAI_KEY = os.getenv("OPENAI_KEY")

if not OPENAI_KEY:
    raise ValueError("OPENAI_KEY not found. Please check your .env file.")

# Example usage
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    openai_api_key=OPENAI_KEY
)
if not os.path.exists("vector_db/index.faiss"):
    print("⚠️ No vector DB found, building one...")
    build_vector_db()


# Load vector DB with embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

# ---------------- INTENT DETECTION ----------------
def detect_intent(inp):
    if m := re.search(r"(cancel|add|track) order\s*#?(\d+)", inp.lower()):
        action, oid = m.groups()
        return action, oid
    return "faq", None

# ---------------- TOOL FUNCTIONS ----------------
def cancel_order(order_id): 
    return f"✅ Order #{order_id} canceled!"

def track_order(order_id): 
    return f"🚚 Order #{order_id} is out for delivery."

def add_item(order_id, item): 
    return f"🍟 Added *{item}* to order #{order_id}."

tools = [
    Tool.from_function(func=cancel_order, name="cancel_order", description="Cancels an order"),
    Tool.from_function(func=track_order, name="track_order", description="Tracks an order"),
    Tool.from_function(func=add_item, name="add_item", description="Adds item to order"),
]

# ---------------- LLM + AGENT ----------------
llm = ChatOpenAI(openai_api_key=OPENAI_KEY, temperature=0)
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

retriever = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=db.as_retriever()
)

# ---------------- QUERY HANDLER ----------------
def handle_query(q):
    intent, oid = detect_intent(q)
    if intent != "faq":
        if intent == "add":
            item = q.split("add")[-1].split("to order")[0].strip()
            return add_item(oid, item)
        return {"cancel": cancel_order, "track": track_order}[intent](oid)
    return retriever.run(q)
