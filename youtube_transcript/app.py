
import os
import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}
.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #60A5FA;
}
.subtitle {
    color: #AAB7C4;
    font-size: 16px;
}
.stChatMessage {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- API KEY ----------------

api_key = st.secrets["GEMINI_API_KEY"]


# ---------------- LOAD RAG SYSTEM ----------------

@st.cache_resource
def create_rag_chain(api_key):

    with open("transcript.txt", "r", encoding="utf-8") as file:
        transcript = file.read()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate(
        template="""
You are a helpful YouTube video assistant.

Answer ONLY using the provided transcript context.

If the answer is not present in the context, say:
"I couldn't find this information in the video transcript."

Give clear and well-structured answers.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.2,
        google_api_key=api_key
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    parallel_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return parallel_chain


# ---------------- INITIALIZE ----------------

st.markdown(
    '<div class="main-title">🎥 YouTube Transcript RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask anything about the video and get answers powered by Google Gemini.</div>',
    unsafe_allow_html=True
)

with st.spinner("Initializing your RAG chatbot..."):
    try:
        rag_chain = create_rag_chain(api_key)
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        st.stop()

st.success("Your video assistant is ready!")


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🎬 Video Assistant")
    st.write("Ask questions based on the uploaded YouTube transcript.")

    st.divider()

    st.subheader("Features")
    st.write("📚 Transcript-based answers")
    st.write("🤖 Gemini AI")
    st.write("🔍 FAISS similarity search")
    st.write("💬 Interactive chat")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- USER INPUT ----------------

question = st.chat_input("Ask something about the video...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching transcript and generating answer..."):
            try:
                answer = rag_chain.invoke(question)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
