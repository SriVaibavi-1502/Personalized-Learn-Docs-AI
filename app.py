import streamlit as st
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
import os
import time


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="LearnDocs AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing in the .env file.")
    st.stop()


# ==================================================
# GEMINI CLIENT
# ==================================================

client = genai.Client(
    api_key=API_KEY
)


# ==================================================
# CHROMADB
# ==================================================

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="gemini_rag_collection"
)


# ==================================================
# SESSION STATE
# ==================================================

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

if "total_characters" not in st.session_state:
    st.session_state.total_characters = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "retrieved_documents" not in st.session_state:
    st.session_state.retrieved_documents = []

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "question" not in st.session_state:
    st.session_state.question = ""


# ==================================================
# HEADER
# ==================================================

st.title("📚 LearnDocs AI")

st.caption(
    "Learn smarter by asking questions directly from your study material. "
    "Powered by Gemini AI and ChromaDB."
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📖 How It Works")

    st.markdown("**1. Upload**")
    st.write("Upload your study PDF.")

    st.markdown("**2. Process**")
    st.write("The PDF is divided into smaller text sections.")

    st.markdown("**3. Search**")
    st.write("ChromaDB finds the most relevant information.")

    st.markdown("**4. Ask**")
    st.write("Type your question about the PDF.")

    st.markdown("**5. Answer**")
    st.write("Gemini generates an answer using the retrieved information.")

    st.divider()

    st.subheader("🎓 Student Tips")

    st.write("💡 Ask specific questions for better answers.")
    st.write("📚 Ask for summaries of your study material.")
    st.write("🔍 Ask about definitions and important points.")
    st.write("✨ Ask for simple explanations.")
    st.write("📝 Use DocuMind AI for quick revision.")


# ==================================================
# PDF TEXT EXTRACTION
# ==================================================

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    text = " ".join(text.split())

    return text


# ==================================================
# TEXT CHUNKING
# ==================================================

def create_chunks(text, chunk_size=1000):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end

    return chunks


# ==================================================
# CREATE GEMINI EMBEDDING
# ==================================================

def create_embedding(text, retries=5):

    for attempt in range(retries):

        try:

            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )

            return response.embeddings[0].values

        except Exception as e:

            error_message = str(e)

            if "503" in error_message or "UNAVAILABLE" in error_message:

                if attempt < retries - 1:

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                else:

                    st.error(
                        "Gemini embedding service is temporarily unavailable. "
                        "Please wait a little and try processing the PDF again."
                    )

                    raise

            else:

                raise


# ==================================================
# STORE CHUNKS IN CHROMADB
# ==================================================

def store_chunks(chunks):

    try:

        chroma_client.delete_collection(
            "gemini_rag_collection"
        )

    except Exception:

        pass

    global collection

    collection = chroma_client.get_or_create_collection(
        name="gemini_rag_collection"
    )

    embeddings = []

    progress = st.progress(0)

    batch_size = 10

    total_chunks = len(chunks)

    for start in range(
        0,
        total_chunks,
        batch_size
    ):

        batch = chunks[
            start:start + batch_size
        ]

        batch_embeddings = None

        for attempt in range(5):

            try:

                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch
                )

                batch_embeddings = [
                    embedding.values
                    for embedding in response.embeddings
                ]

                break

            except Exception as e:

                error_message = str(e)

                if (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                ):

                    if attempt < 4:

                        wait_time = 2 ** attempt

                        time.sleep(wait_time)

                    else:

                        progress.empty()

                        st.error(
                            "Gemini embedding service is temporarily unavailable. "
                            "Please try processing the PDF again after a short wait."
                        )

                        raise

                else:

                    progress.empty()

                    raise

        embeddings.extend(batch_embeddings)

        completed = min(
            start + len(batch),
            total_chunks
        )

        progress.progress(
            completed / total_chunks
        )

    progress.empty()

    ids = [
        f"chunk_{i}"
        for i in range(total_chunks)
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return total_chunks


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(question, context):

    prompt = f"""
You are a helpful AI study assistant.

Answer the student's question using ONLY the information provided in the document context.

Give a clear, accurate and easy-to-understand answer.

IMPORTANT RULES:

- Use complete sentences.
- Use simple English.
- Give a direct answer.
- Do not copy unnecessary text from the document.
- Do not add information that is not in the document.
- If the question asks for multiple points, use bullet points.
- If the question asks for an explanation, use a short paragraph.
- If the question asks for a summary, summarize the important points.
- Keep the answer focused on the question.

If the answer cannot be found in the document, say:

"I could not find the answer in the uploaded document."

DOCUMENT CONTEXT:

{context}

STUDENT QUESTION:

{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# ==================================================
# UPLOAD SECTION
# ==================================================

st.subheader("📄 Upload Your Study Material")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload lecture notes, textbooks, study material or question papers."
)


# ==================================================
# PDF UPLOAD MESSAGE
# ==================================================

if uploaded_file is not None:

    st.success(
        f"📎 {uploaded_file.name} uploaded successfully!"
    )

    if st.button(
        "⚡ Process PDF",
        use_container_width=True
    ):

        with st.spinner(
            "📖 Reading your PDF..."
        ):

            text = extract_text_from_pdf(
                uploaded_file
            )

        if not text.strip():

            st.error(
                "No readable text could be extracted from this PDF."
            )

        else:

            st.info(
                f"📄 Extracted {len(text):,} characters."
            )

            with st.spinner(
                "🧩 Creating document chunks..."
            ):

                chunks = create_chunks(
                    text
                )

            st.info(
                f"🧩 Created {len(chunks)} chunks."
            )

            with st.spinner(
                "🧠 Creating embeddings and storing your PDF..."
            ):

                number_of_chunks = store_chunks(
                    chunks
                )

            st.session_state.pdf_processed = True

            st.session_state.pdf_name = (
                uploaded_file.name
            )

            st.session_state.total_characters = (
                len(text)
            )

            st.session_state.total_chunks = (
                len(chunks)
            )

            st.session_state.retrieved_documents = []

            st.session_state.answer = ""

            st.success(
                f"✅ PDF processed successfully! "
                f"{number_of_chunks} chunks are ready."
            )


# ==================================================
# DOCUMENT INFORMATION
# ==================================================

if st.session_state.get(
    "pdf_processed",
    False
):

    st.divider()

    st.subheader("📊 Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.markdown("### 📄")

            st.markdown("**Uploaded Document**")

            st.write(
                st.session_state.pdf_name
            )

    with col2:

        with st.container(border=True):

            st.markdown("### 📝")

            st.markdown("**Characters**")

            st.write(
                f"{st.session_state.total_characters:,}"
            )

    with col3:

        with st.container(border=True):

            st.markdown("### 🧩")

            st.markdown("**Text Chunks**")

            st.write(
                st.session_state.total_chunks
            )


# ==================================================
# QUESTION SECTION
# ==================================================

if st.session_state.get(
    "pdf_processed",
    False
):

    st.divider()

    st.subheader("💬 Ask Your Question")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What are the main points of this chapter?",
        key="question_input"
    )

    if st.button(
        "🔍 Get Answer",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------
            # SEARCH DOCUMENT
            # ------------------------------------------

            with st.spinner(
                "🔎 Searching your PDF..."
            ):

                question_embedding = create_embedding(
                    question
                )

                results = collection.query(
                    query_embeddings=[
                        question_embedding
                    ],
                    n_results=3
                )

            documents = results[
                "documents"
            ][0]

            st.session_state.retrieved_documents = (
                documents
            )

            context = "\n\n".join(
                documents
            )


            # ------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------

            with st.spinner(
                "🤖 Gemini is preparing your answer..."
            ):

                answer = generate_answer(
                    question,
                    context
                )

            st.session_state.answer = answer


# ==================================================
# ANSWER SECTION
# ==================================================

if st.session_state.get(
    "answer",
    ""
):

    st.divider()

    st.subheader("🤖 Answer")

    with st.container(border=True):

        st.markdown("### 💡 AI Response")

        st.write(
            st.session_state.answer
        )


# ==================================================
# RETRIEVED INFORMATION
# ==================================================

if st.session_state.get(
    "retrieved_documents",
    []
):

    st.divider()

    with st.expander(
        "📖 View Retrieved Information",
        expanded=False
    ):

        for i, doc in enumerate(
            st.session_state.retrieved_documents
        ):

            st.markdown(
                f"**Chunk {i + 1}**"
            )

            st.write(doc)

            if i < len(
                st.session_state.retrieved_documents
            ) - 1:

                st.divider()


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "📚 LearnDocs AI • Gemini AI + ChromaDB • "
    "Built for smarter student learning"
)