import streamlit as st
import sys
import time

st.set_page_config(page_title="Health Check")

st.title("✅ Streamlit Health Check")

st.write("Python version:")
st.code(sys.version)

st.write("If you can see this, Streamlit is running correctly.")

time.sleep(1)
st.success("Health check passed")
