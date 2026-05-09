import streamlit as st
import os
import pickle
import faiss
import numpy as np
import gdown

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Motorsport SAE RAG Assistant",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🏎️ Motorsport SAE RAG Assistant")

st.caption("""
AI Engineering Assistant for:
- Formula SAE
- Baja SAE
- Formula Bharat
- Motorsport Engineering
""")

# =========================================================
# API KEY
# =========================================================

api_key = st.text_input(
    "🔑 Enter Gemini API Key",
    type="password"
)

if not api_key:

    st.warning(
        "Please enter Gemini API Key"
    )

    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# =========================================================
# GOOGLE DRIVE FILE IDS
# =========================================================

FAISS_FILE_ID = "1qDBPT1D2OgAVtDmx6aRfwb46Y3vDlRMs"

METADATA_FILE_ID = "1YS-isyzMNxFgcbVIKcku2KO3pKgaTrwe"

# =========================================================
# DOWNLOAD DATABASE FILES
# =========================================================

if not os.path.exists("final_faiss.index"):

    st.info("📥 Downloading FAISS Database...")

    gdown.download(
        id=FAISS_FILE_ID,
        output="final_faiss.index",
        quiet=False
    )

if not os.path.exists("metadata.pkl"):

    st.info("📥 Downloading Metadata Database...")

    gdown.download(
        id=METADATA_FILE_ID,
        output="metadata.pkl",
        quiet=False
    )

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# =========================================================
# LOAD FAISS DATABASE
# =========================================================

@st.cache_resource
def load_database():

    index = faiss.read_index(
        "final_faiss.index"
    )

    with open("metadata.pkl", "rb") as f:

        metadata = pickle.load(f)

    return index, metadata

index, metadata = load_database()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.success("✅ FAISS Database Loaded")

st.sidebar.success(
    f"📚 Engineering Chunks: {index.ntotal}"
)

# =========================================================
# GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# =========================================================
# RETRIEVAL FUNCTION
# =========================================================

def retrieve_documents(query, top_k=5):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(metadata):

            chunk = metadata[idx]

            if isinstance(chunk, dict):

                if "content" in chunk:

                    text = chunk["content"]

                elif "text" in chunk:

                    text = chunk["text"]

                elif "page_content" in chunk:

                    text = chunk["page_content"]

                else:

                    text = str(chunk)

            else:

                text = str(chunk)

            retrieved_chunks.append(text)

    return retrieved_chunks

# =========================================================
# PROMPT BUILDER
# =========================================================

def build_prompt(query, context):

    prompt = f"""
You are an expert Motorsport Engineering AI Assistant.

You help students participating in:
- Formula SAE
- Baja SAE
- Formula Bharat
- Electric SAE
- Go-Kart competitions

You specialize in:
- Chassis Design
- Suspension Design
- Steering Systems
- Braking Systems
- Aerodynamics
- Vehicle Dynamics
- Powertrain
- CAD / CAE
- FEA / CFD
- Manufacturing
- Materials Engineering

==================================================
IMPORTANT INSTRUCTIONS
==================================================

1. Give practical engineering guidance.

2. Explain concepts in student-friendly language.

3. Use engineering terminology.

4. Give real-world SAE recommendations.

5. Mention:
   - manufacturability
   - safety
   - lightweight design
   - reliability
   - cost effectiveness

6. If formulas are used:
   - explain variables
   - explain meaning
   - explain application

7. Use ONLY the engineering knowledge provided below.

==================================================
ENGINEERING KNOWLEDGE
==================================================

{context}

==================================================
QUESTION
==================================================

{query}

==================================================
ENGINEERING ANSWER
==================================================
"""

    return prompt

# =========================================================
# CHAT HISTORY
# =========================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# =========================================================
# CHAT INPUT
# =========================================================

query = st.chat_input(
    "💬 Ask your Motorsport Engineering question..."
)

# =========================================================
# PROCESS QUERY
# =========================================================

if query:

    with st.spinner(
        "🔍 Searching engineering database..."
    ):

        retrieved_chunks = retrieve_documents(
            query,
            top_k=5
        )

        context = "\n\n".join(
            retrieved_chunks
        )

        prompt = build_prompt(
            query,
            context
        )

    with st.spinner(
        "🤖 Generating engineering answer..."
    ):

        response = llm.invoke(prompt)

        answer = response.content

    st.session_state.chat_history.append(
        (query, answer, retrieved_chunks)
    )

# =========================================================
# DISPLAY CHAT
# =========================================================

for q, a, chunks in st.session_state.chat_history:

    with st.chat_message("user"):

        st.write(q)

    with st.chat_message("assistant"):

        st.write(a)

        with st.expander(
            "📚 Retrieved Engineering Chunks"
        ):

            for i, chunk in enumerate(chunks[:3]):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.write(chunk)

                st.markdown("---")
