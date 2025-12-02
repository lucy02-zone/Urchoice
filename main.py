import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Clean genres text
movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# TF-IDF on genres
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Cosine similarity between movies
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Content-based model ready!")


# ============ Step 3: Collaborative Filtering (User-User Cosine) ============

# Create user-item rating matrix
ratings_matrix = ratings.pivot(index="userId", columns="movieId", values="rating")

# Fill NaN with 0 for similarity computation
ratings_matrix_filled = ratings_matrix.fillna(0)

# Compute user-user similarity
user_similarity = cosine_similarity(ratings_matrix_filled)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=ratings_matrix.index,
    columns=ratings_matrix.index,
)

print("Collaborative (user-based) model ready!")


def get_collab_score(user_id, movie_id):
    """
    Predict rating of 'user_id' for 'movie_id' using
    user-user collaborative filtering (cosine similarity).
    """
    # If movie not in matrix → no info
    if movie_id not in ratings_matrix.columns:
        return 0.0

    # If user not in matrix → no info
    if user_id not in ratings_matrix.index:
        return 0.0

    # Similarity of this user with all users
    sim_scores = user_similarity_df[user_id]

    # All users' ratings for this movie
    movie_ratings = ratings_matrix[movie_id]

    # Consider only users who rated this movie
    mask = movie_ratings.notna()
    if mask.sum() == 0:
        return 0.0

    # Weighted sum: Σ(sim * rating) / Σ(sim)
    numerator = (sim_scores[mask] * movie_ratings[mask]).sum()
    denominator = sim_scores[mask].sum()

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


# ============ Step 4: Hybrid Recommendation System ============

def find_movie(title):
    """Find first movie that contains the given title text."""
    result = movies[movies["title"].str.contains(title, case=False, na=False)]
    if result.empty:
        print("Movie not found! Try again.")
        return None
    print("\nMatching Movies:")
    print(result[["movieId", "title"]].head())
    return int(result.iloc[0]["movieId"])


def hybrid_recommend(movie_title, user_id, alpha=0.5, beta=0.5):
    """
    Hybrid recommendation:
    - content_score from genres similarity
    - collab_score from user-based CF
    final score = alpha * content_score + beta * collab_score
    """
    base_movie_id = find_movie(movie_title)
    if base_movie_id is None:
        return

    # Index of this movie in movies DataFrame
    idx_list = movies.index[movies["movieId"] == base_movie_id].tolist()
    if not idx_list:
        print("Base movie not found in list.")
        return
    idx = idx_list[0]

    # Content-based similarity scores for this movie
    sim_scores = list(enumerate(content_similarity[idx]))
    # Sort by similarity and take top 30 similar (skip itself at index 0)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:31]

    movie_indices = [i[0] for i in sim_scores]
    content_scores = [i[1] for i in sim_scores]

    # Build recommendations DataFrame
    rec_df = movies.iloc[movie_indices][["movieId", "title", "genres"]].copy()
    rec_df["content_score"] = content_scores

    # Collaborative scores for each candidate movie
    collab_scores = []
    for mid in rec_df["movieId"]:
        collab_scores.append(get_collab_score(user_id, mid))
    rec_df["collab_score"] = collab_scores

    # Normalize scores to 0–1
    def min_max(series):
        if series.max() == series.min():
            return series * 0
        return (series - series.min()) / (series.max() - series.min())

    rec_df["content_norm"] = min_max(rec_df["content_score"])
    rec_df["collab_norm"] = min_max(rec_df["collab_score"])

    # Hybrid score
    rec_df["score"] = alpha * rec_df["content_norm"] + beta * rec_df["collab_norm"]

    # Top 10 recommendations
    rec_df = rec_df.sort_values(by="score", ascending=False).head(10)

    print("\nTop 10 Hybrid Recommendations:")
    print(rec_df[["title", "genres", "score"]].to_string(index=False))


# ============ Main ============

if __name__ == "__main__":
    print("\nWelcome to Movie Recommendation System 🎬")

    try:
        user_id = int(input("Enter your User ID (1 to 6040): "))
    except ValueError:
        print("Invalid user ID. Please enter a number.")
        exit()

    movie_title = input("Enter a movie name you like: ")

    hybrid_recommend(movie_title, user_id)
