import numpy
import pandas as pd
import streamlit as st

def main():
    st.title("streamlit-poc")
    st.write("Input info:")
    user_name = st.text_input("Enter your name:", "Player")
    confidence_level = st.slider("Confidence level:", 1, 10, 5)

    st.write(f"Hello {user_name}! Your confidence level is: {confidence_level}")

    data = pd.DataFrame(numpy.random.randn(20, 2), columns=["X", "Y"])
    st.line_chart(data)
    pd.read_parquet

if __name__ == "__main__":
    main()
