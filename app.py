import streamlit as st
import os
import pickle
import faiss
import numpy as np
import gdown

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

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
- Formula SAE (FSAE)
- Formula Bharat
- Baja SAE
- EV Race Cars
- Motorsport Engineering
""")

# =========================================================
# GEMINI API KEY
# =========================================================

api_key = st.text_input(
    "🔑 Enter your Gemini API Key:",
    type="password"
)

if not api_key:

    st.warning(
        "Please enter your Gemini API key to continue"
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
# LOAD FAISS + METADATA
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

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_docs = []

    for idx in indices[0]:

        if idx < len(metadata):

            chunk = metadata[idx]

            # =========================================
            # HANDLE DIFFERENT METADATA STRUCTURES
            # =========================================

            if isinstance(chunk, dict):

                content = ""

                if "content" in chunk:
                    content = chunk["content"]

                elif "text" in chunk:
                    content = chunk["text"]

                elif "page_content" in chunk:
                    content = chunk["page_content"]

                else:
                    content = str(chunk)

                metadata_info = {
                    "file_name": chunk.get(
                        "file_name",
                        "Engineering Source"
                    ),
                    "page": chunk.get(
                        "page",
                        "-"
                    )
                }

            else:

                content = str(chunk)

                metadata_info = {
                    "file_name": "Engineering Source",
                    "page": "-"
                }

            retrieved_docs.append(
                Document(
                    page_content=content,
                    metadata=metadata_info
                )
            )

    return retrieved_docs

# =========================================================
# PROMPT BUILDER
# =========================================================

def build_prompt(query, docs):

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an expert Motorsport Engineering AI Assistant.

You help engineering students participating in:

- Formula SAE (FSAE)
- Formula Bharat
- Baja SAE
- Electric SAE
- Go-Kart competitions

You specialize in:
- Chassis Design
- Suspension Systems
- Steering Systems
- Braking Systems
- Aerodynamics
- Vehicle Dynamics
- Powertrain
- Manufacturing
- CAD / CAE
- FEA / CFD
- Materials Engineering

==================================================
IMPORTANT INSTRUCTIONS
==================================================

1. Give practical engineering answers.

2. Explain concepts in student-friendly language.

3. Use proper engineering terminology.

4. If formulas are needed:
   - explain variables
   - explain engineering meaning
   - explain why formula is used

5. Give real-world SAE design guidance.

6. Mention:
   - manufacturability
   - safety
   - reliability
   - lightweight design
   - cost-effectiveness

7. If applicable provide:
   - material suggestions
   - manufacturing methods
   - design recommendations
   - common mistakes to avoid

8. Use ONLY the provided engineering knowledge.

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
# USER INPUT
# =========================================================

query = st.chat_input(
    "💬 Ask your Motorsport Engineering question..."
)

# =========================================================
# PROCESS QUERY
# =========================================================

if query:

    with st.spinner(
        "🔍 Searching engineering knowledge base..."
    ):

        docs = retrieve_documents(
            query,
            top_k=5
        )

        prompt = build_prompt(
            query,
            docs
        )

    with st.spinner(
        "🤖 Generating engineering answer..."
    ):

        response = llm.invoke(prompt)

        answer = response.content

    st.session_state.chat_history.append(
        (query, answer, docs)
    )

# =========================================================
# DISPLAY CHAT
# =========================================================

for q, a, docs in st.session_state.chat_history:

    with st.chat_message("user"):

        st.write(q)

    with st.chat_message("assistant"):

        st.write(a)

        st.markdown(
            "### 📚 Engineering Sources"
        )

        for doc in docs[:3]:

            st.write(
                f"• {doc.metadata.get('file_name', 'Engineering Source')} "
                f"(Page {doc.metadata.get('page', '-')})"
            )
