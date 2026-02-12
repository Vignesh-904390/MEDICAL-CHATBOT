import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ===============================
# CONFIG
# ===============================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "gale-medical-index")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY missing in .env file")

# ===============================
# SET ENV VARIABLE (IMPORTANT)
# ===============================
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ===============================
# LOAD EMBEDDING MODEL
# ===============================
print("🧠 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===============================
# CONNECT TO PINECONE
# ===============================
print("📡 Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

# LOAD EXISTING INDEX (FIXED)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=embeddings
)

print("✅ Connected to medical database")

# ===============================
# RETRIEVE CONTEXT
# ===============================
def retrieve_context(question: str, k: int = 3) -> List[Document]:
    return vectorstore.similarity_search(question, k=k)

# ===============================
# MAIN CHAT FUNCTION
# ===============================
def ask_gale_bot(question: str) -> str:

    docs = retrieve_context(question)

    if not docs:
        return "No medical information found in database."

    context = "\n\n".join(doc.page_content for doc in docs)

    final_answer = f"""
🩺 Medical Information:

{context}

⚠️ Educational purpose only.
Consult doctor for diagnosis.
"""
    return final_answer
