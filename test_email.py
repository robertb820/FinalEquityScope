import streamlit as st

# Initialize session state
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# Email Registration
st.title("Test Email Registration")
if not st.session_state.user_email:
    with st.form("email_form"):
        email = st.text_input("Enter your email to get started:", "")
        submit = st.form_submit_button("Submit")
        if submit and email:
            st.session_state.user_email = email
            st.experimental_rerun()
    st.warning("Please enter your email to proceed.")
    st.stop()
else:
    st.success(f"Registered email: {st.session_state.user_email}")
    st.write("This is a basic website to test email registration.")