import streamlit as st
import faiss
import pickle
import numpy as np
import os
import gdown

from google import genai

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Motorsport RAG Agent",
    page_icon="🏎️",
    layout="wide"
)

# =====================================================
# GOOGLE DRIVE FILE IDS
# =====================================================

FAISS_FILE_ID = "1qDBPT1D2OgAVtDmx6aRfwb46Y3vDlRMs"
METADATA_FILE_ID = "1YS-isyzMNxFgcbVIKcku2KO3pKgaTrwe"

# =====================================================
# DOWNLOAD DATABASE FILES
# =====================================================

if not os.path.exists("final_faiss.index"):
    with st.spinner("Downloading FAISS Database..."):
        gdown.download(
            id=FAISS_FILE_ID,
            output="final_faiss.index",
            quiet=False
        )

if not os.path.exists("metadata.pkl"):
    with st.spinner("Downloading Metadata..."):
        gdown.download(
            id=METADATA_FILE_ID,
            output="metadata.pkl",
            quiet=False
        )

# =====================================================
# LOAD DATABASE
# =====================================================

@st.cache_resource
def load_rag_database():
    index = faiss.read_index("final_faiss.index")

    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

index, metadata = load_rag_database()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🏎️ Motorsport RAG Agent")

st.sidebar.success("FAISS Database Loaded")
st.sidebar.success(f"Vectors Loaded: {index.ntotal}")

# =====================================================
# GEMINI API SECTION
# =====================================================

st.sidebar.subheader("Gemini API Key")

api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

connect_btn = st.sidebar.button("Connect API")

api_connected = False

if connect_btn:

    try:
        client = genai.Client(api_key=api_key)

        test_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="hello"
        )

        st.sidebar.success("Gemini Connected Successfully")
        api_connected = True

    except Exception as e:
        st.sidebar.error(f"API Error: {e}")

# =====================================================
# MAIN UI
# =====================================================

st.title("🏎️ Motorsport Engineering RAG Assistant")

st.markdown("""
Ask engineering questions related to:

- Motorsport
- Formula Racing
- Aerodynamics
- Suspension
- Chassis
- Powertrain
- Mechanical Design
- Vehicle Dynamics
- Materials
- CAD & Manufacturing
""")

query = st.text_area(
    "Ask Your Engineering Question",
    height=150
)

ask_button = st.button("Generate Answer")

# =====================================================
# EMBEDDING + RETRIEVAL
# =====================================================

def get_query_embedding(client, text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    embedding = response.embeddings[0].values

    return np.array([embedding], dtype=np.float32)

# =====================================================
# GENERATE RESPONSE
# =====================================================

if ask_button:

    if not api_key:
        st.warning("Please Enter Gemini API Key")
        st.stop()

    try:

        client = genai.Client(api_key=api_key)

        with st.spinner("Generating Engineering Response..."):

            # =========================================
            # QUERY EMBEDDING
            # =========================================

            query_embedding = get_query_embedding(
                client,
                query
            )

            # =========================================
            # VECTOR SEARCH
            # =========================================

            k = 5

            distances, indices = index.search(
                query_embedding,
                k
            )

            retrieved_chunks = []

            for idx in indices[0]:

                if idx < len(metadata):

                    chunk = metadata[idx]

                    if isinstance(chunk, dict):

                        retrieved_chunks.append(
                            chunk.get("text", "")
                        )

                    else:
                        retrieved_chunks.append(str(chunk))

            # =========================================
            # CONTEXT BUILDING
            # =========================================

            context = "\n\n".join(retrieved_chunks)

            # =========================================
            # FINAL PROMPT
            # =========================================

            final_prompt = f"""
You are an expert Motorsport and Mechanical Engineering AI Assistant.

Use the retrieved engineering knowledge below to answer the user's question.

========================
ENGINEERING KNOWLEDGE
========================

{context}

========================
USER QUESTION
========================

{query}

========================
ANSWER
========================
"""

            # =========================================
            # GEMINI RESPONSE
            # =========================================

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt
            )

            # =========================================
            # DISPLAY ANSWER
            # =========================================

            st.subheader("Engineering Answer")

            st.write(response.text)

            # =========================================
            # SHOW RETRIEVED CHUNKS
            # =========================================

            with st.expander("Retrieved Engineering Chunks"):

                for i, chunk in enumerate(retrieved_chunks):

                    st.markdown(f"### Chunk {i+1}")

                    st.write(chunk[:1500])

    except Exception as e:

        st.error(f"Error: {e}")
