import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="NEA",
)

# navigation/option
with st.sidebar:
   selected = option_menu(
        menu_title="Main Menu",  
        options=["Home","Insertion","Extraction"], 
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

if selected == "Insertion":
    st.write("# NEA ")
    st.write(
    """
    **Masih Kosong**.
    """
    )

if selected == "Extraction":
    st.write("# NEA ")
    st.write(
    """
    **Ini Juga Masih Kosong**.
    """
    )
