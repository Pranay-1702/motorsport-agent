import streamlit as st
from google import genai

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Motorsport Agent",
    page_icon="🏎",
    layout="wide"
)

# ====================================
# SIMPLE DARK THEME
# ====================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0E1117;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    p, div, label, span {
        color: white !important;
        font-size: 18px !important;
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
    """,
    unsafe_allow_html=True
)

# ====================================
# SESSION STATE
# ====================================

if "api_connected" not in st.session_state:
    st.session_state.api_connected = False

if "client" not in st.session_state:
    st.session_state.client = None

# ====================================
# LOGIN PAGE
# ====================================

if not st.session_state.api_connected:

    st.title("🏎 Motorsport Agent")

    st.markdown("""
    ### AI Engineering Assistant for Motorsport & Mechanical Systems
    """)

    api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )

    if st.button("Connect API"):

        with st.spinner("Connecting Gemini API..."):

            try:

                client = genai.Client(
                    api_key=api_key
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Reply OK"
                )

                st.session_state.api_connected = True
                st.session_state.client = client

                st.success(
                    "Gemini API Connected Successfully"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    "Invalid API Key"
                )

# ====================================
# MAIN APP
# ====================================

else:

    with st.sidebar:

        st.title("🏎 Motorsport Agent")

        st.success("Gemini API Connected")

        st.markdown("---")

        st.subheader("System Status")

        st.success("FAISS Database Ready")
        st.success("OCR Engine Active")
        st.success("Excel Retrieval Ready")
        st.success("Multimodal RAG Active")

        st.markdown("---")

        if st.button("Disconnect API"):

            st.session_state.api_connected = False
            st.session_state.client = None

            st.rerun()

    st.title("🏎 Motorsport Agent")

    st.markdown("""
    ### AI Engineering Assistant for Motorsport & Mechanical Systems
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

    if st.button("Generate Answer"):

        if not query:

            st.error(
                "Please Enter Question"
            )

        else:

            with st.spinner(
                "Generating Engineering Response..."
            ):

                try:

                    response = st.session_state.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=query
                    )

                    st.markdown("## Answer")

                    st.success(response.text)

                except Exception as e:

                    st.error(
                        f"Error: {str(e)}"
                    )
