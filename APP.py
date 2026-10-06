import streamlit as st
from PIL import Image
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import tempfile
import base64
import requests


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f5f7fb;
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #1e293b;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* Section heading */
    .section-title {
        font-size: 24px;
        font-weight: 650;
        color: #1e293b;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        padding: 22px 15px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        min-height: 145px;
        margin-bottom: 20px;
    }

    .feature-icon {
        font-size: 35px;
        margin-bottom: 8px;
    }

    .feature-title {
        font-size: 18px;
        font-weight: 650;
        color: #1e293b;
    }

    .feature-description {
        font-size: 14px;
        color: #64748b;
        margin-top: 7px;
    }

    /* Upload cards */
    .upload-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Information box */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border-left: 5px solid #6366f1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 20px;
        margin-bottom: 25px;
        color: #334155;
        line-height: 1.7;
    }

    /* Image */
    img {
        border-radius: 15px !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        margin-bottom: 10px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding: 25px;
        margin-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 Multimodal RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Text + Image Understanding using RAG and AI'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FEATURE CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📄</div>
        <div class="feature-title">Text Encoder</div>
        <div class="feature-description">
            Converts PDF text into embeddings for RAG retrieval.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🖼️</div>
        <div class="feature-title">Image Encoder</div>
        <div class="feature-description">
            Understands uploaded images using LLaVA.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">Llama AI</div>
        <div class="feature-description">
            Combines the retrieved information and generates answers.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📂 Upload Your Data</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="upload-card">',
        unsafe_allow_html=True
    )

    st.subheader("📄 Upload PDF")

    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        '<div class="upload-card">',
        unsafe_allow_html=True
    )

    st.subheader("🖼️ Upload Image")

    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# PDF PROCESSING
# =========================================================

retrieved_text = ""
vector_db = None

if pdf_file is not None:

    with st.spinner("📄 Processing PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_pdf:

            temp_pdf.write(
                pdf_file.getvalue()
            )

            pdf_path = temp_pdf.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        # Split PDF into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(
            documents
        )

        # Text encoder
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text:v1.5"
        )

        # Create vector database
        vector_db = Chroma.from_documents(
            chunks,
            embeddings
        )

    st.success(
        f"✅ PDF processed successfully! "
        f"{len(chunks)} text chunks created."
    )


# =========================================================
# IMAGE DISPLAY
# =========================================================

if image_file is not None:

    st.markdown(
        '<div class="section-title">🖼️ Uploaded Image</div>',
        unsafe_allow_html=True
    )

    image = Image.open(image_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=600
    )

    st.success("✅ Image uploaded successfully!")


# =========================================================
# INFORMATION BOX
# =========================================================

st.markdown("""
<div class="info-box">

<b>How this chatbot works:</b><br><br>

📄 <b>PDF</b> → Text Encoder → Vector Database → Relevant Text<br>

🖼️ <b>Image</b> → LLaVA Vision Model → Image Understanding<br>

📚 <b>Retrieved Information + Image Understanding</b>
→ Llama AI → Final Answer

</div>
""", unsafe_allow_html=True)


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "💬 Ask something about your PDF or image..."
)


# =========================================================
# QUESTION PROCESSING
# =========================================================

if question:

    # Show user's question
    with st.chat_message("user"):
        st.write(question)

    retrieved_text = ""
    image_description = ""

    # -----------------------------------------------------
    # PDF RETRIEVAL
    # -----------------------------------------------------

    if vector_db is not None:

        with st.spinner("🔎 Searching the PDF..."):

            results = vector_db.similarity_search(
                question,
                k=3
            )

            retrieved_text = "\n\n".join(
                doc.page_content
                for doc in results
            )


    # -----------------------------------------------------
    # IMAGE UNDERSTANDING USING LLAVA
    # -----------------------------------------------------

    if image_file is not None:

        with st.spinner("👁️ LLaVA is analyzing the image..."):

            image_bytes = image_file.getvalue()

            image_base64 = base64.b64encode(
                image_bytes
            ).decode("utf-8")

            vision_prompt = f"""
Look carefully at the uploaded image.

Answer the user's question based ONLY on
what you can actually understand from the image.

Identify:
- Important objects
- Text visible in the image
- Labels
- Diagrams
- Charts
- Shapes
- Visual relationships
- Other relevant visual information

Do not give a generic answer.

User question:
{question}
"""

            try:

                response = requests.post(
                    "http://localhost:11434/api/chat",

                    json={
                        "model": "llava:latest",

                        "messages": [
                            {
                                "role": "user",
                                "content": vision_prompt,
                                "images": [
                                    image_base64
                                ]
                            }
                        ],

                        "stream": False
                    },

                    timeout=300
                )

                if response.status_code == 200:

                    vision_data = response.json()

                    image_description = (
                        vision_data["message"]["content"]
                    )

                else:

                    image_description = (
                        "Unable to analyze the image. "
                        "Please check whether Ollama and "
                        "LLaVA are running."
                    )

            except Exception as e:

                image_description = (
                    f"Image analysis error: {str(e)}"
                )


    # -----------------------------------------------------
    # FINAL PROMPT FOR LLAMA
    # -----------------------------------------------------

    final_prompt = f"""
You are the final AI assistant for a multimodal RAG chatbot.

Answer the user's question using the information
provided below.

USER QUESTION:
{question}


INFORMATION RETRIEVED FROM PDF:
{retrieved_text}


INFORMATION UNDERSTOOD FROM IMAGE:
{image_description}


Instructions:

1. If the question is about the PDF,
   use the PDF information.

2. If the question is about the image,
   use the image information.

3. If both PDF and image information are relevant,
   combine them.

4. Give a clear and simple answer.

5. Do not invent information that is not available
   in the provided PDF or image.

6. If the information is not available,
   clearly say that it is not available.
"""


# =========================================================
# LLAMA FINAL ANSWER
# =========================================================

    with st.spinner("🧠 Llama is generating the answer..."):

        try:

            llama_response = requests.post(
                "http://localhost:11434/api/chat",

                json={
                    "model": "llama3.2:3b",

                    "messages": [
                        {
                            "role": "user",
                            "content": final_prompt
                        }
                    ],

                    "stream": False
                },

                timeout=300
            )

            if llama_response.status_code == 200:

                llama_data = llama_response.json()

                final_answer = (
                    llama_data["message"]["content"]
                )

            else:

                final_answer = (
                    "Unable to generate the final answer. "
                    "Please check whether Ollama is running."
                )

        except Exception as e:

            final_answer = (
                f"Error generating answer: {str(e)}"
            )


    # =====================================================
    # DISPLAY FINAL ANSWER
    # =====================================================

    with st.chat_message("assistant"):

        st.markdown("### 🤖 Answer")

        st.write(final_answer)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    🤖 Multimodal RAG Chatbot |
    Text Encoder + Image Understanding + Llama AI
</div>
""", unsafe_allow_html=True)
