import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG-Based Chatbot")
st.subheader("Using Text and Image Encoders")

st.write(
    "Upload a PDF and an image, then ask questions about your content."
)


# ==========================================
# 2. MODELS
# ==========================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:v1.5"
)

llm = ChatOllama(
    model="gpt-oss:20b-cloud"
)


# ==========================================
# 3. FILE UPLOAD
# ==========================================

st.sidebar.header("📂 Upload Files")

pdf_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

image_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# 4. DISPLAY IMAGE
# ==========================================

if image_file is not None:

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image_file,
        caption="Uploaded Image",
        width=500
    )


# ==========================================
# 5. PROCESS PDF
# ==========================================

if pdf_file is not None:

    st.info("Processing PDF...")

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            pdf_file.getvalue()
        )

        pdf_path = temp_file.name


    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()


    # ======================================
    # 6. SPLIT TEXT
    # ======================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )


    # ======================================
    # 7. CREATE TEXT EMBEDDINGS
    # ======================================

    st.info("Creating text embeddings...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    st.success(
        "PDF processed successfully! ✅"
    )


    # ======================================
    # 8. CHAT INPUT
    # ======================================

    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    if question:

        # ==================================
        # 9. RETRIEVE INFORMATION
        # ==================================

        retriever = vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

        results = retriever.invoke(
            question
        )


        # ==================================
        # 10. CREATE CONTEXT
        # ==================================

        context = "\n\n".join(
            document.page_content
            for document in results
        )


        # ==================================
        # 11. CREATE PROMPT
        # ==================================

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful RAG chatbot.

Answer the user's question using only
the information available in the provided context.

If the answer is not available in the context,
say:

"The information is not available in
the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
        )


        formatted_prompt = prompt.format(
            context=context,
            question=question
        )


        # ==================================
        # 12. GENERATE ANSWER
        # ==================================

        with st.spinner("Generating answer..."):

            response = llm.invoke(
                formatted_prompt
            )


        # ==================================
        # 13. DISPLAY CHAT
        # ==================================

        st.chat_message(
            "user"
        ).write(question)

        st.chat_message(
            "assistant"
        ).write(response.content)


        # ==================================
        # 14. SHOW RETRIEVED INFORMATION
        # ==================================

        with st.expander(
            "📚 Retrieved Information"
        ):

            for i, document in enumerate(
                results
            ):

                st.write(
                    f"### Result {i + 1}"
                )

                st.write(
                    document.page_content
                )


# ==========================================
# 15. IMAGE INFORMATION
# ==========================================

if image_file is not None:

    st.subheader("🖼️ Image Encoder")

    st.write(
        "Image uploaded successfully."
    )

    st.write(
        "The image encoder module will process "
        "the visual information in the next stage."
    )