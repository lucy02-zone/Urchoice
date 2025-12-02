import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ============ Step 1: Load Data ============

movies_path = "data/movies.dat"
ratings_path = "data/ratings.dat"

movies = pd.read_csv(
    movies_path,
    sep="::",
    engine="python",
    names=["movieId", "title", "genres"],
)

ratings = pd.read_csv(
    ratings_path,
    sep="::",
    engine="python",
    names=["userId", "movieId", "rating", "timestamp"],
)

print("Data Loaded Successfully!")
print("Movies:", movies.shape, "Ratings:", ratings.shape)


# ============ Step 2: Content-Based Similarity (Genres) ============

movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Content-based model ready!")


# ============ Step 3: Collaborative Filtering (Item-Item Cosine) ============

ratings_matrix = ratings.pivot(index="movieId", columns="userId", values="rating")
ratings_matrix_filled = ratings_matrix.fillna(0)

# Load saved similarity if exists
if os.path.exists("item_similarity.npy"):
    print("Loading similarity matrix from file...")
    item_similarity = np.load("item_similarity.npy")
else:
    print("Computing similarity (takes 1-2 minutes)...")
    item_similarity = cosine_similarity(ratings_matrix_filled)
    np.save("item_similarity.npy", item_similarity)

item_similarity_df = pd.DataFrame(
    item_similarity, index=ratings_matrix.index, columns=ratings_matrix.index
)

print("Collaborative model ready!")


def get_collab_score(movie_id, user_id):
    if movie_id not in ratings_matrix.index or user_id not in ratings_matrix.columns:
        return 0.0

    sim_scores = item_similarity_df.loc[movie_id]
    user_ratings = ratings_matrix[user_id]

    mask = user_ratings.notna()
    if mask.sum() == 0:
        return 0.0

    numerator = (sim_scores[mask] * user_ratings[mask]).sum()
    denominator = sim_scores[mask].sum()

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


# ============ Step 4: Hybrid Recommendation System ============

def find_movie(title):
    results = movies[movies["title"].str.contains(title, case=False, na=False)]
    if results.empty:
        return None
    return int(results.iloc[0]["movieId"])


def hybrid_recommend(movie_title, user_id, alpha=0.5, beta=0.5, return_df=False):
    movie_id = find_movie(movie_title)
    if movie_id is None:
        raise ValueError("Movie not found!")

    idx = movies.index[movies["movieId"] == movie_id][0]
    sim_scores = list(enumerate(content_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:31]

    movie_indices = [i[0] for i in sim_scores]
    content_scores = [i[1] for i in sim_scores]

    rec_df = movies.iloc[movie_indices][["movieId", "title", "genres"]].copy()
    rec_df["content_score"] = content_scores

    rec_df["collab_score"] = [
        get_collab_score(mid, user_id) for mid in rec_df["movieId"]
    ]

    def min_max(x):
        if x.max() == x.min(): return x * 0
        return (x - x.min()) / (x.max() - x.min())

    rec_df["content_norm"] = min_max(rec_df["content_score"])
    rec_df["collab_norm"] = min_max(rec_df["collab_score"])

    rec_df["score"] = alpha * rec_df["content_norm"] + beta * rec_df["collab_norm"]
    rec_df = rec_df.sort_values("score", ascending=False).head(10)

    if return_df:
        return rec_df[["title", "genres", "score"]]

    print("\nTop 10 Hybrid Recommendations:")
    print(rec_df[["title", "genres", "score"]].to_string(index=False))


# ============ Command Line Run ============

if __name__ == "__main__":
    print("\n🎬 Welcome to Movie Recommendation System 🎬")
    user = int(input("Enter your User ID: "))
    movie = input("Enter a movie you like: ")
    hybrid_recommend(movie, user)
