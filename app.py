import streamlit as st
import faiss
import pickle
import numpy as np
import os
import gdown
import google.generativeai as genai

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Motorsport SAE RAG Assistant",
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

    with st.spinner("Downloading FAISS database..."):

        gdown.download(
            id=FAISS_FILE_ID,
            output="final_faiss.index",
            quiet=False
        )

if not os.path.exists("metadata.pkl"):

    with st.spinner("Downloading metadata database..."):

        gdown.download(
            id=METADATA_FILE_ID,
            output="metadata.pkl",
            quiet=False
        )

# =====================================================
# LOAD DATABASE
# =====================================================

index = faiss.read_index("final_faiss.index")

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🏎️ Motorsport SAE RAG")

st.sidebar.success("FAISS Database Loaded")
st.sidebar.success(f"Vectors Loaded: {index.ntotal}")

# =====================================================
# GEMINI API KEY
# =====================================================

st.sidebar.subheader("Gemini API Key")

api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

api_connected = False

# =====================================================
# CONNECT API
# =====================================================

if st.sidebar.button("Connect API"):

    try:

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content("hello")

        st.session_state["api_key"] = api_key

        st.sidebar.success("Gemini API Connected Successfully")

        api_connected = True

    except Exception as e:

        st.sidebar.error(f"API Error: {e}")

# =====================================================
# RESTORE SESSION
# =====================================================

if "api_key" in st.session_state:

    genai.configure(
        api_key=st.session_state["api_key"]
    )

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    api_connected = True

# =====================================================
# MAIN TITLE
# =====================================================

st.title("🏎️ Motorsport SAE RAG Assistant")

st.markdown("""
### Ask questions related to:

- Chassis Design
- Suspension
- Braking Systems
- Aerodynamics
- Steering
- Vehicle Dynamics
- Powertrain
- CAD & CAE
- Manufacturing
- Materials
- FEA & CFD
""")

# =====================================================
# QUESTION INPUT
# =====================================================

question = st.text_area(
    "Ask Your Engineering Question",
    height=150
)

# =====================================================
# EMBEDDING FUNCTION
# =====================================================

def get_query_embedding(text):

    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )

    embedding = result["embedding"]

    return np.array([embedding], dtype=np.float32)

# =====================================================
# SEARCH FUNCTION
# =====================================================

def search_documents(query, top_k=5):

    query_embedding = get_query_embedding(query)

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(metadata):

            chunk = metadata[idx]

            if isinstance(chunk, dict):

                text = chunk.get("text", "")

            else:

                text = str(chunk)

            retrieved_chunks.append(text)

    return retrieved_chunks

# =====================================================
# GENERATE ANSWER
# =====================================================

def generate_answer(question, context):

    engineering_prompt = f"""
You are an expert Motorsport Engineering AI Assistant.

You help students participating in:

- Formula SAE
- Baja SAE
- Formula Student

You provide:
- Practical engineering guidance
- Vehicle design support
- Chassis recommendations
- Suspension tuning advice
- Braking system knowledge
- Aerodynamic concepts
- Manufacturing guidance
- CAD / CAE / CFD support
- FEA recommendations

IMPORTANT:
- Use ONLY the provided engineering knowledge.
- Give detailed and student-friendly answers.
- Explain concepts clearly.
- Give practical real-world SAE guidance.

==================================================
ENGINEERING KNOWLEDGE
==================================================

{context}

==================================================
QUESTION
==================================================

{question}

==================================================
ENGINEERING ANSWER
==================================================
"""

    response = model.generate_content(
        engineering_prompt
    )

    return response.text

# =====================================================
# GENERATE BUTTON
# =====================================================

if st.button("Generate Engineering Answer"):

    if not api_connected:

        st.error("Please connect Gemini API first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        # =============================================
        # SEARCH DATABASE
        # =============================================

        with st.spinner("Searching engineering database..."):

            retrieved_chunks = search_documents(
                question,
                top_k=5
            )

            context = "\n\n".join(retrieved_chunks)

        # =============================================
        # GENERATE ANSWER
        # =============================================

        with st.spinner("Generating engineering answer..."):

            answer = generate_answer(
                question,
                context
            )

        # =============================================
        # DISPLAY ANSWER
        # =============================================

        st.subheader("Engineering Answer")

        st.write(answer)

        # =============================================
        # SHOW RETRIEVED CHUNKS
        # =============================================

        with st.expander("Retrieved Engineering Chunks"):

            for i, chunk in enumerate(retrieved_chunks):

                st.markdown(f"### Chunk {i+1}")

                st.write(chunk)

                st.markdown("---")
