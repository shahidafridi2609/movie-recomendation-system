import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

st.set_page_config(page_title="Movie Recommendation System", page_icon="Images/animals.gif")

data = pd.read_excel("netflix_data.xlsx")

# Modify 'Title' to ensure it's of string type
data['Title'] = data['Title'].astype(str)
data['Title'] = [title.lower() for title in data['Title']]

# Demographic Filtering
C = data["IMDb Score"].mean()
m = data["IMDb Votes"].quantile(0.9)
popular_movies = data.copy().loc[data["IMDb Votes"] >= m]

def weighted_rating(x, C=C, m=m):
    v = x["IMDb Votes"]
    R = x["IMDb Score"]
    return (v / (v + m) * R) + (m / (v + m) * C)

popular_movies["Score"] = popular_movies.apply(weighted_rating, axis=1)
popular_movies = popular_movies.sort_values('Score', ascending=False)

def plot():
    top_rated_and_popular = popular_movies.head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top_rated_and_popular["Title"], top_rated_and_popular["Score"], align="center", color="skyblue")
    ax.invert_yaxis()
    ax.set_title("Top 10 Popular Movies")
    ax.set_xlabel("Score")
    return fig

# Content-Based Filtering
data["Languages"] = data["Languages"].fillna("")
data["Genre"] = data["Genre"].fillna("")
data["Series or Movie"] = data["Series or Movie"].fillna("")

data['Features'] = data['Languages'] + ' ' + data['Genre'] + ' ' + data['Series or Movie']
data['Features'] = data['Features'].values.astype('U')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Features'])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data["Title"]).drop_duplicates()

def get_recommendations(title):
    title = title.lower()
    if all(title not in movie_title.lower() for movie_title in data["Title"]):
        return "Movie not found"
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = data['Title'].iloc[movie_indices]
    return recommended_movies

def main():
    image = st.image('Images/net.png', use_column_width=True)
    
    st.title("Movie Recommendation System")
    choice = st.sidebar.selectbox(
        "Choose an option:",
        ("Home", "Demographic Filtering", "Content-Based Filtering", "Exit")
    )

    if choice == "Home":
        st.markdown(
            "<div style='text-align: center;'>"
            "<h3>Welcome to the Movie Recommendation System</h3>"
            "</div>",
            unsafe_allow_html=True
        )

        image = st.image('Images/animals.gif', use_column_width=True)
    elif choice == "Demographic Filtering":
        st.markdown("Top Rated and Popular Movies:")
        st.dataframe(data[["Title", "IMDb Votes", "View Rating", "IMDb Score"]].head(10))
        st.markdown("Top Rated and Popular Movies in bar-plot:")
        st.pyplot(plot())

    elif choice == "Content-Based Filtering":
        st.subheader("Movie Recommendations")
        movie_title = st.text_input("Enter a movie title for recommendations:")
        if movie_title:
            recommendations = get_recommendations(movie_title)
            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                for idx, movie in enumerate(recommendations, start=1):
                    st.write(f"{idx}. {movie}")

    elif choice == "Exit":
        st.write("Goodbye!")

if __name__ == "__main__":
    main()
