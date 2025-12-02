Hybrid Movie Recommendation System (MovieLens 1M)

A personalized movie recommendation system built using Content-Based Filtering + Item-Item Collaborative Filtering on the MovieLens 10M dataset.

🚀 This hybrid approach improves accuracy by combining:

Movie similarity based on genres (TF-IDF + Cosine Similarity)

User preference patterns using item-based collaborative filtering

🧠 Features

✔ Recommend movies similar to a movie user likes
✔ Personalized predictions based on user’s previous ratings
✔ Works with large dataset (10M+ ratings)
✔ Hybrid model = better recommendation performance
✔ Command-line based interaction (UI upgrade available)

🛠️ Tech Stack
Component	Technology
Programming	Python
ML Libraries	scikit-learn, pandas, numpy
Dataset	MovieLens 10M (GroupLens Research)
📂 Folder Structure
movie-recommender/
│
├── data/
│   ├── movies.dat
│   ├── ratings.dat
│
├── main.py
├── README.md
└── venv/ (optional virtual environment)

📦 Dataset Download

MovieLens 10M Official Page:
🔗 https://grouplens.org/datasets/movielens/10m/

Extract → Move these into data/ folder:

movies.dat

ratings.dat

⚙️ Installation

Create & activate virtual environment (optional but recommended):

python -m venv venv
venv\Scripts\activate   # Windows


Install required packages:

pip install pandas numpy scikit-learn

▶️ How to Run

Inside your project folder:

python main.py


Sample input:

Welcome to Movie Recommendation System 🎬
Enter your User ID (1 to 69878): 10
Enter a movie name you like: Toy Story


Output example:

Top 10 Hybrid Recommendations:
Movie Title | Genres | Hybrid Score
...

🧮 How the Model Works
Step	Description
1️⃣ Content-Based Filtering	Convert genres → TF-IDF → cosine similarity
2️⃣ Collaborative Filtering	Movie rating vectors → cosine similarity
3️⃣ Hybrid Scoring	Combined weighted score from both models
4️⃣ Top-N Results	Return best recommended movies

Formula:

Hybrid Score = 0.5 * Content Score + 0.5 * Collaborative Score

🚀 Future Enhancements

✔ Streamlit UI with movie posters
✔ Model performance evaluation (RMSE, Precision@K)
✔ Save similarity matrices to reduce load time
✔ Add more metadata like actors, directors, summaries

📜 License

Dataset © GroupLens Research
Code free to use for educational purposes 🔓

🙌 Acknowledgements

MovieLens Dataset by GroupLens

scikit-learn team for ML tools