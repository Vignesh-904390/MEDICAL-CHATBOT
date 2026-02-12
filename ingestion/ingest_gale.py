import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config import PINECONE_API_KEY, PINECONE_INDEX

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# ---------------- Pinecone ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete old index (3072 dim)
if PINECONE_INDEX in [i.name for i in pc.list_indexes()]:
    print("🗑 Deleting old Pinecone index...")
    pc.delete_index(PINECONE_INDEX)
    time.sleep(5)

# Create new index (384 dim)
pc.create_index(
    name=PINECONE_INDEX,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

while not pc.describe_index(PINECONE_INDEX).status["ready"]:
    time.sleep(2)

print("✅ Pinecone index ready (384 dims)")

# ---------------- Load PDF ----------------
PDF_PATH = os.path.join(BASE_DIR, "data", "gale_encyclopedia_of_medicine.pdf")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# ---------------- Split ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=200
)
docs = splitter.split_documents(documents)

print(f"📄 Pages: {len(documents)}")
print(f"✂️ Chunks: {len(docs)}")

# ---------------- Local embeddings ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- Store ----------------
PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=PINECONE_INDEX,
    pinecone_api_key=PINECONE_API_KEY
)


print("🎉 Gale Encyclopedia ingested successfully (LOCAL embeddings)")
