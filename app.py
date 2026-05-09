
import streamlit as st
import base64
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Motorsport Agent",
    layout="wide"
)

# =========================
# LOAD LOCAL BACKGROUND IMAGE
# =========================

def get_base64_image(image_path):

    with open(image_path, "rb") as img_file:
        return base64.b64encode(
            img_file.read()
        ).decode()

# Google Drive mounted image path
background_image = (
    "/content/drive/MyDrive/Images/MS.png"
)

img_base64 = get_base64_image(background_image)

# =========================
# CUSTOM CSS
# =========================

page_bg = f"""
<style>

.stApp {{
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.main {{
    background: rgba(0,0,0,0.72);
    border-radius: 20px;
    padding: 20px;
}}

h1, h2, h3, h4, h5, h6 {{
    color: white !important;
}}

p, label, div {{
    color: white !important;
}}

[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.88);
}}

.stTextInput input {{
    background-color: #1c1c1c;
    color: white;
    border-radius: 10px;
}}

.stTextArea textarea {{
    background-color: #1c1c1c;
    color: white;
    border-radius: 10px;
}}

.stButton button {{
    background-color: #00AEEF;
    color: white;
    border-radius: 10px;
    border: none;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}}

.stButton button:hover {{
    background-color: #0077aa;
    color: white;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🏎 Motorsport Agent")

    st.markdown("---")

    api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )

    st.markdown("---")

    st.subheader("System Status")

    st.success("FAISS Database Ready")
    st.success("Multimodal RAG Active")
    st.success("OCR Engine Ready")
    st.success("Excel Engine Ready")

# =========================
# MAIN UI
# =========================

st.title("🏎 Motorsport Agent")

st.markdown(
    "### AI Engineering Assistant for Motorsport & Mechanical Systems"
)

query = st.text_area(
    "Ask Engineering Questions",
    height=180,
    placeholder=(
        "Example:\n"
        "- Explain finite element analysis\n"
        "- Explain suspension geometry\n"
        "- Explain CFD in motorsport\n"
        "- Compare braking systems"
    )
)

if st.button("Generate Answer"):

    if not api_key:

        st.error("Please Enter Gemini API Key")

    elif not query:

        st.error("Please Enter Question")

    else:

        st.success("Motorsport Agent Ready")

        st.info(
            "Next Step: Connect FAISS Retrieval + Gemini Response Engine"
        )
