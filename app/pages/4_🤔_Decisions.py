import streamlit as st
import os

st.set_page_config(page_title="Decisions", page_icon="🤔")

html_file_path = os.path.join("build", "html", "index.html")


if os.path.exists(html_file_path):
    with open(html_file_path, 'r', encoding='utf-8') as html_file:
        html_content = html_file.read()
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.error("HTML file not found. Please check the file path.")
