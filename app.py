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

    with st.spinner("Downloading Metadata File..."):

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

connect_btn = st.sidebar.button("Connect API")

if connect_btn:

    try:

        client = genai.Client(api_key=api_key)

        test_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="hello"
        )

        st.sidebar.success("Gemini Connected Successfully")

    except Exception as e:

        st.sidebar.error(f"API Error: {e}")

# =====================================================
# MAIN UI
# =====================================================

st.title("🏎️ Motorsport Engineering RAG Assistant")

st.markdown("""
### AI Engineering Mentor for:

- Formula Student (FSAE)
- Formula Bharat
- Baja SAE
- Go-Karts
- EV Race Cars
- Motorsport Vehicles
- Mechanical Design Projects

Ask questions related to:

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

query = st.text_area(
    "Ask Your Engineering Question",
    height=180
)

generate_btn = st.button("Generate Engineering Answer")

# =====================================================
# EMBEDDING FUNCTION
# =====================================================

def get_query_embedding(client, text):

    response = client.models.embed_content(
        model="models/embedding-001",
        contents=text
    )

    embedding = response.embeddings[0].values

    return np.array([embedding], dtype=np.float32)

# =====================================================
# GENERATE RESPONSE
# =====================================================

if generate_btn:

    if not api_key:

        st.warning("Please Enter Gemini API Key")
        st.stop()

    try:

        client = genai.Client(api_key=api_key)

        with st.spinner("Analyzing Engineering Knowledge Base..."):

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

            # =========================================
            # RETRIEVE CHUNKS
            # =========================================

            retrieved_chunks = []

            for idx in indices[0]:

                if idx < len(metadata):

                    chunk = metadata[idx]

                    chunk_text = ""

                    # ---------------------------------
                    # CASE 1 : DICTIONARY
                    # ---------------------------------

                    if isinstance(chunk, dict):

                        if "text" in chunk:
                            chunk_text = chunk["text"]

                        elif "content" in chunk:
                            chunk_text = chunk["content"]

                        elif "page_content" in chunk:
                            chunk_text = chunk["page_content"]

                        elif "chunk" in chunk:
                            chunk_text = chunk["chunk"]

                        else:
                            chunk_text = str(chunk)

                    # ---------------------------------
                    # CASE 2 : STRING
                    # ---------------------------------

                    elif isinstance(chunk, str):

                        chunk_text = chunk

                    # ---------------------------------
                    # CASE 3 : OTHER
                    # ---------------------------------

                    else:

                        chunk_text = str(chunk)

                    if chunk_text.strip():

                        retrieved_chunks.append(chunk_text)

            # =========================================
            # CONTEXT BUILDING
            # =========================================

            context = "\n\n".join(retrieved_chunks[:5])

            if context.strip() == "":

                context = "No engineering context retrieved."

            # =========================================
            # FINAL PROMPT
            # =========================================

            final_prompt = f"""
You are an expert Motorsport, Formula Student (FSAE), Baja SAE, and Mechanical Engineering Mentor AI.

Your role is to help engineering students design, analyze, manufacture, and improve SAE competition vehicles.

You must answer like an experienced SAE mentor and race vehicle engineer.

==================================================
IMPORTANT RESPONSE RULES
==================================================

1. Explain answers in simple engineering language suitable for students.

2. Give practical and industry-standard solutions.

3. Focus on:
   - FSAE
   - Formula Bharat
   - Baja SAE
   - Go-karts
   - Electric SAE vehicles
   - Motorsport engineering

4. When applicable include:
   - Recommended materials
   - Design considerations
   - Manufacturing methods
   - Safety considerations
   - Performance impact
   - Advantages and disadvantages
   - Real-world SAE practices

5. If the question is related to:
   - Brakes
   - Suspension
   - Chassis
   - Aerodynamics
   - Steering
   - Powertrain
   - Vehicle dynamics
   - CAD
   - Manufacturing

   then provide engineering reasoning step-by-step.

6. If formulas are needed:
   - Explain formulas clearly
   - Explain variables
   - Explain why formula is used

7. Always encourage good engineering practices:
   - lightweight design
   - reliability
   - manufacturability
   - safety
   - serviceability

8. If retrieved context is weak or incomplete:
   - still answer using motorsport engineering knowledge
   - do NOT say "context is empty"

9. Keep answers educational, practical, and detailed.

10. If possible, provide:
   - Recommended SAE design approaches
   - Typical competition practices
   - Cost-effective solutions for students
   - Common mistakes to avoid

==================================================
RETRIEVED ENGINEERING KNOWLEDGE
==================================================

{context}

==================================================
STUDENT QUESTION
==================================================

{query}

==================================================
ENGINEERING ANSWER
==================================================
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

            with st.expander("Retrieved Engineering Knowledge"):

                if len(retrieved_chunks) == 0:

                    st.warning("No chunks retrieved from database.")

                else:

                    for i, chunk in enumerate(retrieved_chunks):

                        st.markdown(f"### Chunk {i+1}")

                        st.write(chunk[:2000])

    except Exception as e:

        st.error(f"Error: {e}")
