import streamlit as st

# Initialize session state for device mode
if "device_mode" not in st.session_state:
    st.session_state.device_mode = "computer"

# Sidebar with device toggle
st.sidebar.title("Settings")
st.sidebar.markdown("**Debug:** Device toggle should appear below this.")
device_mode = st.sidebar.radio("Select Device Mode:", ["Computer", "Mobile"], index=0 if st.session_state.device_mode == "computer" else 1)
st.session_state.device_mode = device_mode.lower()
st.sidebar.markdown("**Debug:** Device toggle rendered successfully.")

# Main page
st.write(f"Current Device Mode: {st.session_state.device_mode}")