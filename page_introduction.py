import streamlit as st

def page_introduction():
    
    # Space so that 'About' box-text is lower
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    
    st.markdown("<h2 style='text-align: center;'> Welcome To </h2>", 
                unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'> Distribution Analyser</h1>", 
                unsafe_allow_html=True)
     

    st.info("""
            There are two main features: \n
            - Get Data 
            - Fit distributions  
            $←$ To start playing with the app, select an option on the 
            left sidebar.
            """)
    
    return