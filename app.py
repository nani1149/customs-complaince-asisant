import chainlit as cl
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


os.environ["AZURE_OPENAI_API_KEY"] = "<azure key>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<azure subscription id>.openai.azure.com/"


# Load environment variables
load_dotenv()

# Initialize Azure OpenAI chat model
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_type="azure",
    temperature=0,
)

# Initialize embeddings model
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Load FAISS vector index
FAISS_INDEX_DIR = "faiss_index"
db = FAISS.load_local(
    FAISS_INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True,
)

# # 1. Define your prompt template
template = """Your are an Harmonised Tariff Expert.Answer the question in your own words as truthfully as possible from the context given to you.
If you do not know the answer to the question, simply respond with "I don't know. Can you ask another question".
If questions are asked where there is no relevant context available, simply respond with "I don't know. Please ask a question relevant to the documents"
Context: {context}


{chat_history}
Human: {question}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"], template=template
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # This tells memory what output to store
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={'prompt': prompt},
    output_key="answer",  # This tells the chain what the main output is
)

# On chat start
@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Welcome to the **ETS Assistant**!\n\n📚 Ask me anything related to your uploaded documents."
    ).send()

# On user message
@cl.on_message
async def main(message: cl.Message):
    res = qa_chain(message.content)

    answer = res["answer"]  # instead of res["result"]
    sources = res.get("source_documents", [])

    # Format source display with page and content
    source_texts = "\n\n".join(
        [
            f"📄 **Page {doc.metadata.get('page', '?')} of {doc.metadata.get('source', '?')}**\n"
            f"```text\n{doc.page_content.strip()[:1000]}\n```"
            for doc in sources
        ]
    )

    final_response = f"""
🎯 **Answer:**

{answer}

---

📚 **Sources:**

{source_texts}
"""
    await cl.Message(content=final_response).send()