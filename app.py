import streamlit as st
import faiss
import pickle
import numpy as np
import os
import gdown

from google import genai

# ==========================================
# GOOGLE DRIVE FILE IDS
# ==========================================

FAISS_FILE_ID = "1qDBPT1D2OgAVtDmx6aRFwb46Y3vDLRMs"

METADATA_FILE_ID = "1YS-isyzMNxFgcbVIKcku2KO3pKgaTrwe"

# ==========================================
# DOWNLOAD VECTOR DATABASE
# ==========================================

if not os.path.exists("final_faiss.index"):

    with st.spinner("Downloading FAISS Database..."):

        gdown.download(
            id=FAISS_FILE_ID,
            output="final_faiss.index",
            quiet=False
        )

if not os.path.exists("metadata.pkl"):

    with st.spinner("Downloading Metadata Database..."):

        gdown.download(
            id=METADATA_FILE_ID,
            output="metadata.pkl",
            quiet=False
        )

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Motorsport Engineering RAG",
    page_icon="🏎",
    layout="wide"
)

# ==========================================
# CUSTOM DARK UI
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

h1, h2, h3, h4, h5 {
    color: white !important;
}

p, div, label, span {
    color: white !important;
}

section[data-testid="stSidebar"] {
    background-color: #161A23;
}

.stTextInput input {
    background-color: #1E2430 !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #00AEEF;
}

.stTextArea textarea {
    background-color: #1E2430 !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #00AEEF;
    font-size: 18px !important;
}

.stButton button {
    background-color: #00AEEF !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD VECTOR DATABASE
# ==========================================

@st.cache_resource
def load_database():

    index = faiss.read_index(
        "final_faiss.index"
    )

    with open(
        "metadata.pkl",
        "rb"
    ) as f:

        metadata = pickle.load(f)

    return index, metadata

index, metadata = load_database()

# ==========================================
# SESSION STATE
# ==========================================

if "connected" not in st.session_state:
    st.session_state.connected = False

if "client" not in st.session_state:
    st.session_state.client = None

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🏎 Motorsport RAG")

    st.success(
        f"FAISS Loaded: {index.ntotal} vectors"
    )

    st.success("Engineering PDFs Loaded")
    st.success("Excel Knowledge Loaded")
    st.success("Mechanical RAG Active")

    st.markdown("---")

    api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )

    if st.button("Connect Gemini API"):

        try:

            client = genai.Client(
                api_key=api_key
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Reply OK"
            )

            st.session_state.client = client
            st.session_state.connected = True

            st.success(
                "Gemini Connected Successfully"
            )

        except Exception as e:

            st.error(
                "Invalid API Key"
            )

# ==========================================
# MAIN PAGE
# ==========================================

st.title("🏎 AI Motorsport Engineering RAG")

st.markdown("""
### Engineering Knowledge Assistant using:
- FAISS Vector Search
- Engineering PDFs
- Motorsport Knowledge
- Excel Cost Reports
- Gemini AI
""")

query = st.text_area(
    "Ask Engineering Questions",
    height=220,
    placeholder="""
Examples:
- Best material for gokart chassis
- Explain finite element analysis
- Explain Formula Student aerodynamics
- Difference between MIG and TIG welding
"""
)

# ==========================================
# RETRIEVAL FUNCTION
# ==========================================

def retrieve_chunks(query, top_k=5):

    embed_response = st.session_state.client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )

    query_embedding = np.array(
        [embed_response.embeddings[0].values],
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(metadata):

            retrieved_chunks.append(
                metadata[idx]
            )

    return retrieved_chunks

# ==========================================
# GENERATE ENGINEERING ANSWER
# ==========================================

if st.button("Generate Engineering Answer"):

    if not st.session_state.connected:

        st.error(
            "Please Connect Gemini API"
        )

    elif query.strip() == "":

        st.error(
            "Please Enter Question"
        )

    else:

        with st.spinner(
            "Searching Engineering Knowledge Base..."
        ):

            chunks = retrieve_chunks(query)

            context = "\n\n".join(
                [
                    chunk["content"][:2000]
                    for chunk in chunks
                ]
            )

        with st.spinner(
            "Generating Engineering Response..."
        ):

            final_prompt = f"""
You are an expert Motorsport and Mechanical Engineering AI Assistant.

Use ONLY the engineering context below.

ENGINEERING CONTEXT:
{context}

QUESTION:
{query}

Give detailed technical engineering answer.
"""

            response = st.session_state.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt
            )

            st.markdown("## Engineering Answer")

            st.success(
                response.text
            )

            st.markdown("---")

            st.markdown(
                "## Retrieved Engineering Sources"
            )

            for i, chunk in enumerate(chunks):

                st.markdown(
                    f"""
### Source {i+1}

File:
{chunk.get('file_name', 'Unknown')}

Page:
{chunk.get('page', 'N/A')}

Preview:
{chunk['content'][:700]}
"""
                )
