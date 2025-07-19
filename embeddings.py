import os
from PyPDF2 import PdfReader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Embedding config
embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("embed_end_point"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Directory to store FAISS index
FAISS_DIR = "faiss_index"

# Load PDF and convert to chunks
def load_pdf_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            for chunk in text.split('\n\n'):
                chunk = chunk.strip()
                if len(chunk) > 30:
                    doc = Document(
                        page_content=chunk,
                        metadata={"source": os.path.basename(pdf_path), "page": page_num}
                    )
                    chunks.append(doc)
    return chunks

# Add chunks to FAISS vectorstore
def add_to_vector_db(chunks):
    if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        db = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(FAISS_DIR)

# Process multiple PDFs
def process_multiple_pdfs(pdf_folder):
    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing {filename}...")
            chunks = load_pdf_chunks(pdf_path)
            all_chunks.extend(chunks)
    add_to_vector_db(all_chunks)
    print(f"\n✅ Added {len(all_chunks)} chunks from all PDFs in '{pdf_folder}'")

# Example usage
process_multiple_pdfs("data4")
