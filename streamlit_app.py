import streamlit as st

st.set_page_config(
    page_title="Homework Landing Page",
    layout="wide",
    initial_sidebar_state= 'expanded'

)
#welcome_page = st.Page('')
HW1 = st.Page('HW/HW1.py', title="Homework 1")
HW2 = st.Page('HW/HW2.py', title = "Homework 2", default=True)

pg = st.navigation([HW1, HW2])
st.set_page_config(page_title='Homework Manager')
pg.run()