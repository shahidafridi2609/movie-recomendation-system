import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_excel("netflix_data.xlsx")

# Modify 'Title' to ensure it's of string type
data['Title'] = data['Title'].astype(str)
data['Title']=[title.lower() for title in data['Title'] ]

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
    plt.figure(figsize=(12, 6))
    plt.barh(top_rated_and_popular["Title"], top_rated_and_popular["Score"], align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Popular Movies")
    plt.xlabel("Score")

# Call the plot function to display the chart

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words="english")
data["Languages"] = data["Languages"].fillna("")  # Handle missing values in the "Summary" column
data["Genre"] = data["Genre"].fillna("")  # Handle missing values in the "Summary" column
data["Series or Movie"] = data["Series or Movie"].fillna("")  # Handle missing values in the "Summary" column

data['Features'] = data['Languages'] + ' ' + data['Genre'] + ' ' + data['Series or Movie']
data['Features']=data['Features'].values.astype('U')
# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Features'])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data["Title"]).drop_duplicates()

def get_recommendations(title):
    # Convert the user input to lowercase
    title = title.lower()

    # Check if the user input is not in any movie titles
    if all(title not in movie_title.lower() for movie_title in data["Title"]):
        return "Movie not found"

    # Find the movie that matches the user input

    # Get the index of the matched title
    idx = indices[title]
    # Calculate the cosine similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies= data['Title'].iloc[movie_indices]
    
    return recommended_movies

def main():
    print("Welcome to the Movie Recommendation System")
    while True:
        print("\nChoose an option:")
        print("1. Demographic Filtering (Top Rated and Popular Movies)")
        print("2. Content-Based Filtering (Movie Recommendations)")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            print("Top Rated and Popular Movies:")
            print(data[["Title", "IMDb Votes", "View Rating", "IMDb Score"]].head(10))
            plot()  # Calling the plot function to display the chart
        elif choice == '2':
            movie_title = input("Enter a movie title for recommendations: ")
            recommendations = get_recommendations(movie_title)
            print(f"Recommended Movies for '{movie_title}':")
            if isinstance(recommendations, str):
                print(recommendations)
            else:
                for idx, movie in enumerate(recommendations, start=1):
                    print(f"{idx}. {movie}")
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please choose a valid option.")

if __name__ == "__main__":
    main()
