import streamlit as st
import os
import pickle
import faiss
import numpy as np
import gdown

from sentence_transformers import SentenceTransformer
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Motorsport SAE RAG",
    layout="wide"
)

st.title("🏎️ Motorsport SAE RAG Assistant")

st.markdown("""
### AI Assistant for:
- Formula Student
- Baja SAE
- Formula Bharat
- Motorsport Engineering
""")


# =========================================================
# GOOGLE DRIVE FILE IDS
# =========================================================

FAISS_FILE_ID = "1qDBPT1D2OgAVtDmx6aRfwb46Y3vDlRMs"
METADATA_FILE_ID = "1YS-isyzMNxFgcbVIKcku2KO3pKgaTrwe"


# =========================================================
# DOWNLOAD DATABASE FILES
# =========================================================

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


# =========================================================
# LOAD DATABASE
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
# SIDEBAR STATUS
# =========================================================

st.sidebar.success("✅ FAISS Database Loaded")

st.sidebar.success(
    f"📚 Engineering Chunks: {len(metadata)}"
)

st.sidebar.info(
    f"📐 Embedding Dimension: {index.d}"
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    # 384 dimension
    if index.d == 384:

        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # 768 dimension
    elif index.d == 768:

        model_name = "sentence-transformers/all-mpnet-base-v2"

    else:

        model_name = "sentence-transformers/all-mpnet-base-v2"

    st.sidebar.success(
        f"🧠 Embedding Model Loaded"
    )

    return SentenceTransformer(model_name)


embedding_model = load_embedding_model()


# =========================================================
# GEMINI API KEY
# =========================================================

st.markdown("## 🔑 Enter Gemini API Key")

api_key = st.text_input(
    "Gemini API Key",
    type="password"
)

if not api_key:

    st.warning("Please enter Gemini API Key")
    st.stop()


# =========================================================
# CONNECT GEMINI
# =========================================================

try:

    client = genai.Client(
        api_key=api_key
    )

    st.sidebar.success("✅ Gemini API Connected")

except Exception as e:

    st.error(f"API Error: {e}")
    st.stop()


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

def retrieve_documents(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

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
# PROMPT TEMPLATE
# =========================================================

def build_prompt(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an expert Motorsport Engineering AI Assistant specialized in:

- Formula Student
- Baja SAE
- Formula Bharat
- EV Race Cars
- Chassis Design
- Suspension
- Braking Systems
- Steering Systems
- Vehicle Dynamics
- Aerodynamics
- Powertrain
- CAD & CAE
- FEA & CFD
- Manufacturing
- Materials Engineering

Your purpose is to help engineering students design and build SAE competition vehicles.

Use ONLY the provided engineering knowledge context.

If information exists in the context:
- Explain clearly
- Give engineering reasoning
- Include formulas if needed
- Give practical SAE-level recommendations
- Mention design tradeoffs
- Keep answers student-friendly but technically strong

If information is not available in context:
Say:
"Relevant engineering data was not found in the knowledge base."

==================================================

ENGINEERING KNOWLEDGE:

{context}

==================================================

QUESTION:
{question}

==================================================

ANSWER:
"""

    return prompt


# =========================================================
# QUESTION INPUT
# =========================================================

st.markdown("## Ask Your Engineering Question")

question = st.text_area(
    "",
    placeholder="Example: Which braking system is best for Formula Student EV?"
)


# =========================================================
# GENERATE ANSWER
# =========================================================

if st.button("Generate Engineering Answer"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating engineering answer..."):

            try:

                # =========================================
                # RETRIEVE CHUNKS
                # =========================================

                retrieved_chunks = retrieve_documents(
                    question,
                    top_k=5
                )

                # =========================================
                # BUILD PROMPT
                # =========================================

                prompt = build_prompt(
                    question,
                    retrieved_chunks
                )

                # =========================================
                # GEMINI RESPONSE
                # =========================================

                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt
                )

                answer = response.text

                # =========================================
                # DISPLAY ANSWER
                # =========================================

                st.markdown("## Engineering Answer")

                st.write(answer)

                # =========================================
                # SHOW RETRIEVED CHUNKS
                # =========================================

                with st.expander("Retrieved Engineering Knowledge"):

                    for i, chunk in enumerate(retrieved_chunks):

                        st.markdown(f"### Chunk {i+1}")

                        st.write(chunk[:2000])

            except Exception as e:

                st.error(f"Error: {e}")
