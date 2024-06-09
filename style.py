import streamlit as st

def apply_styling():
    st.markdown(
        """
        <style>
        .main {
            background-color: #2828222;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #ADFF2F;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
