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
    page_title="Motorsport SAE RAG Assistant",
    layout="wide"
)

# =====================================================
# GOOGLE DRIVE FILE IDS
# =====================================================

FAISS_FILE_ID = "1qDBPT1D2OgAVtDmx6aRfwb46Y3vDlRMs"
METADATA_FILE_ID = "1YS-isyzMNxFgcbVIKcku2KO3pKgaTrwe"

# =====================================================
# DOWNLOAD FILES FROM GOOGLE DRIVE
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
# LOAD FAISS DATABASE
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
# GEMINI API SECTION
# =====================================================

st.sidebar.subheader("Gemini API Key")

api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

api_connected = False

# =====================================================
# CONNECT BUTTON
# =====================================================

if st.sidebar.button("Connect API"):

    try:

        client = genai.Client(api_key=api_key)

        test = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="hello"
        )

        st.session_state["api_key"] = api_key

        st.sidebar.success("Gemini API Connected Successfully")

        api_connected = True

    except Exception as e:

        st.sidebar.error(f"API Error: {e}")

# =====================================================
# RESTORE SESSION
# =====================================================

if "api_key" in st.session_state:

    client = genai.Client(
        api_key=st.session_state["api_key"]
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
# USER QUESTION
# =====================================================

question = st.text_area(
    "Ask Your Engineering Question",
    height=150
)

# =====================================================
# EMBEDDING FUNCTION
# =====================================================

def get_query_embedding(client, text):

    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=[text]
    )

    embedding = response.embeddings[0].values

    return np.array([embedding], dtype=np.float32)

# =====================================================
# SEARCH DOCUMENTS
# =====================================================

def search_documents(client, query, top_k=5):

    query_embedding = get_query_embedding(
        client,
        query
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

                text = chunk.get("text", "")

            else:

                text = str(chunk)

            retrieved_chunks.append(text)

    return retrieved_chunks

# =====================================================
# GENERATE ANSWER
# =====================================================

def generate_answer(client, question, context):

    engineering_prompt = f"""
You are an expert AI Motorsport Engineering Assistant.

You help students participating in:
- Formula SAE
- Baja SAE
- Formula Student

Your job is to provide:
- Engineering explanations
- Practical design guidance
- Real-world motorsport knowledge
- SAE vehicle design suggestions
- Manufacturing recommendations
- Component selection guidance
- CAD / CAE / FEA / CFD support
- Vehicle dynamics explanations

IMPORTANT RULES:
- Use ONLY the engineering knowledge provided below.
- If information is insufficient, clearly say so.
- Explain answers in student-friendly language.
- Give practical engineering guidance.
- Be detailed and technical when necessary.

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=engineering_prompt
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
        # SEARCH
        # =============================================

        with st.spinner("Searching engineering database..."):

            retrieved_chunks = search_documents(
                client,
                question,
                top_k=5
            )

            context = "\n\n".join(retrieved_chunks)

        # =============================================
        # GENERATE RESPONSE
        # =============================================

        with st.spinner("Generating engineering answer..."):

            answer = generate_answer(
                client,
                question,
                context
            )

        # =============================================
        # SHOW ANSWER
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
