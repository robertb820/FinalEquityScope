import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import requests
import time
import logging
import json
import socket

# Email prompt
if "user_email" not in st.session_state:
    st.session_state.user_email = None

if not st.session_state.user_email:
    st.warning("Please provide your email to access the app.")
    email = st.text_input("Enter your email:", "")
    if st.button("Submit"):
        if email:
            st.session_state.user_email = email
            st.experimental_rerun()
    st.stop()

st.write(f"Welcome, {st.session_state.user_email}!")
st.write("The app is working correctly!")