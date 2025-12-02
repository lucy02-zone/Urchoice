import streamlit as st
from main import hybrid_recommend

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Personalized movie suggestions using Hybrid ML Model")

movie_name = st.text_input("Enter a movie name:")
user_id = st.number_input("Enter user ID:", step=1, min_value=1)

if st.button("Recommend"):
    try:
        df = hybrid_recommend(movie_name, int(user_id), return_df=True)
        st.success("Recommended Movies 🎯")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error: {e}")
