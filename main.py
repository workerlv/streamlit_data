import streamlit as st
from OpenCVBasicFunctions import main_cv2
from Git import main_git

select_box = st.selectbox(
    "Pick category",
    ("CV2", "Git")
)

st.header(select_box)

option_functions = {
    "CV2": main_cv2.MainCV,
    "Git": main_git.GitInfo,
}

option_functions[select_box]()



