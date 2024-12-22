import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="NEA",
)

# navigation/option
with st.sidebar:
   selected = option_menu(
        menu_title="Main Menu",  
        options=["Home","Encryption","Decryption"], 
        icons=["house", "record-circle"],  
        menu_icon="cast",  # optional
        default_index=0,  # optional         
)

if selected == "Home":
    st.write("# NEA ")
    st.write(
    """
    **Protect Your Document Here**.
    """
    )
    
    st.caption("Created by **Zein Rivo**")

if selected == "Encryption":
    st.write("# FACETERA ")
    st.write(
    """
    **Multi-Task Cascaded Convolutional Neural Networks (MTCNN)**.
    """
    )

if selected == "Decryption":
    st.write("# FACETERA ")
    st.write(
    """
    **Multi-Task Cascaded Convolutional Neural Networks (MTCNN)**.
    """
    )
