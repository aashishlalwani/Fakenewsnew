import streamlit as st
from predict import predict_news

st.title("Fake News Detection System")

st.write("""
Enter the text of a news article below to check if it is likely to be real or fake.
This tool uses a machine learning model to analyze the text.
""")

user_input = st.text_area("News Article Text", "Enter text here...", height=250)

if st.button("Analyze"):
    if user_input and user_input != "Enter text here...":
        with st.spinner("Analyzing..."):
            prediction = predict_news(user_input)
            st.subheader("Analysis Result")
            if prediction == "REAL":
                st.success("This article appears to be REAL.")
            else:
                st.error("This article appears to be FAKE.")
    else:
        st.warning("Please enter some text to analyze.")
