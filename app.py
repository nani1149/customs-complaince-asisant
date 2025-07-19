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

import chainlit as cl

LANGUAGE_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de"
}


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
    actions = [
        cl.Action(name=lang_code, value=lang_code, label=lang_name)
        for lang_name, lang_code in LANGUAGE_MAP.items()
    ]

    await cl.Message(content="🌍 Please select your preferred language:").send()

    msg = cl.AskActionMessage(
        content="Choose your language from the dropdown below 👇",
        actions=actions,
    )
    res = await msg.send()

    # Save language in session
    cl.user_session.set("user_lang", res)
    await cl.Message(
        content=f"✅ Language set to **{[k for k, v in LANGUAGE_MAP.items() if v == res][0]}**.\n\nYou can now ask your questions!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    user_lang = cl.user_session.get("user_lang", "en")

    # Translate to English if not already
    if user_lang != "en":
        translated_question = azure_translate(message.content, to_lang="en", from_lang=user_lang)
    else:
        translated_question = message.content

    # Run the QA chain
    res = qa_chain(translated_question)
    answer = res["answer"]
    sources = res.get("source_documents", [])

    # Translate answer back to user's language
    if user_lang != "en":
        answer = azure_translate(answer, to_lang=user_lang, from_lang="en")

    # Format sources
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

"""
    await cl.Message(content=final_response).send()
