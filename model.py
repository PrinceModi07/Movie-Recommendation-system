import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Print the first few rows and column names to verify
print("Movies DataFrame Columns:", movies.columns)
print(movies.head())

print("Ratings DataFrame Columns:", ratings.columns)
print(ratings.head())

# Preprocessing
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convert user-item matrix to a sparse matrix format
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

# Ensure k is less than the smallest dimension of the matrix
k = min(user_item_matrix_sparse.shape) - 1 if 50 > min(user_item_matrix_sparse.shape) else 50

# Matrix Factorization
U, sigma, Vt = svds(user_item_matrix_sparse, k=k)
sigma = np.diag(sigma)

# Predicted Ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns)


# Recommendation Function
def recommend_movies(user_id, num_recommendations=5):
    user_row_number = user_id - 1  # UserID starts at 1, not 0
    sorted_user_predictions = predicted_ratings_df.iloc[user_row_number].sort_values(ascending=False)

    user_data = ratings[ratings.userId == user_id]
    print("User Data Columns:", user_data.columns)
    print("Movies Data Columns:", movies.columns)

    user_full = (user_data.merge(movies, how='left', left_on='movieId', right_on='movieId')
                 .sort_values(['rating'], ascending=False))
    print("User Full DataFrame:", user_full.head())

    # Reset index of sorted_user_predictions for merging
    sorted_user_predictions = sorted_user_predictions.reset_index().rename(columns={user_row_number: 'Predictions'})
    print("Sorted User Predictions DataFrame:", sorted_user_predictions.head())

    recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])]
                       .merge(sorted_user_predictions, how='left', on='movieId')
                       .sort_values('Predictions', ascending=False)
                       .iloc[:num_recommendations, :-1])
    print("Recommendations DataFrame:", recommendations.head())

    return user_full, recommendations


if __name__ == "__main__":
    # Test the function with a valid user ID
    already_rated, predictions = recommend_movies(1)
    print("Already Rated Movies:", already_rated)
    print("Predictions:", predictions)
