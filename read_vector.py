from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = "<azure key>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<azure subscription id>.openai.azure.com/"


# Initialize embedding model
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment=os.environ["OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# Function to load and view chunks
def view_chunks_from_faiss(index_path="faiss_index", limit=10):
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search("example", k=limit)  # dummy search to get documents
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:500])  # preview first 500 characters
        print("Metadata:", doc.metadata)
        print()

# Example usage
view_chunks_from_faiss()
