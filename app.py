import streamlit as st
import pandas as pd
from main import hybrid_recommend

st.set_page_config(page_title="Movie Recommender", layout="wide")

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None


# Function to display detail page
def show_movie_details(movie):
    st.markdown("<h2 style='text-align:center;'>🎬 {}</h2>".format(movie["title"]), unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(movie["poster"], width=250)
    
    st.markdown(f"<p style='text-align:center;'>Genres: {movie['genres']}</p>", unsafe_allow_html=True)

    # Convert score (0-1) to 1-5 star scale
    star_rating = int(round(movie["score"] * 5))
    stars = "⭐" * star_rating + "☆" * (5 - star_rating)

    st.markdown(f"<h3 style='text-align:center;'>{stars}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>Hybrid Score: {round(movie['score'], 3)}</p>", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("⬅ Back to Recommendations"):
        st.session_state.selected_movie = None
        st.rerun()



# Main UI Section
st.title("🎞 Hybrid Movie Recommendation System")
st.write("Get personalized recommendations instantly!")


movie_name = st.text_input("Enter a movie you like:")
user_id = st.number_input("Enter your User ID:", step=1, min_value=1, max_value=70000)


if st.button("Recommend 🎯"):
    try:
        df = hybrid_recommend(movie_name, int(user_id), return_df=True)
        st.session_state.recommendations = df
        st.session_state.selected_movie = None

    except Exception as e:
        st.error(f"❌ {e}")


# If user clicks a movie → show details page
if st.session_state.selected_movie is not None:
    movie = st.session_state.selected_movie
    show_movie_details(movie)

elif "recommendations" in st.session_state:
    df = st.session_state.recommendations

    st.success("Top Picks For You 🍿")

    cols = st.columns(3)
    col_index = 0

    for _, movie in df.iterrows():
        with cols[col_index]:
            st.image(movie["poster"], width=200)
            if st.button(f"Select_{movie['title']}", key=movie['title']):
                st.session_state.selected_movie = movie
                st.rerun()

            st.markdown(f"**{movie['title']}**")
            st.caption(movie["genres"])

        col_index = (col_index + 1) % 3
        if col_index == 0:
            cols = st.columns(3)
