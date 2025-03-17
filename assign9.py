#question-1
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances

# Load and preprocess the movie data
def load_movie_data(file_path):
    # Read CSV file
    movies_df = pd.read_csv('movie.csv')
    
    # Create one-hot encoding for genres
    # Assuming genres are in a column 'Genre' and separated by '|'
    genres = movies_df['genres'].str.get_dummies('|')
    
    # Combine genres with ratings
    features = pd.concat([genres, movies_df['rating']], axis=1)
    
    return movies_df, features

# Function to find k nearest movies
def find_similar_movies(movie_features, movie_index, k):
    # Calculate distances from the target movie to all other movies
    distances = euclidean_distances([movie_features.iloc[movie_index]], movie_features)
    
    # Get indices of k nearest movies (excluding the movie itself)
    similar_indices = np.argsort(distances[0])[1:k+1]
    
    return similar_indices

# Main recommendation function
def recommend_movies(movie_title, movies_df, movie_features, k=5):
    # Find the index of the input movie
    try:
        movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        return "Movie not found in database!"
    
    # Find similar movies
    similar_indices = find_similar_movies(movie_features, movie_idx, k)
    
    # Get the similar movies' information
    recommendations = movies_df.iloc[similar_indices][['title', 'genres', 'rating']]
    
    return recommendations

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    movies_df, movie_features = load_movie_data('movie.csv')
    
    # Test the recommendation system
    movie_title = "Toy Story (1995)"
    k = 5
    
    print(f"Recommendations for {movie_title}:")
    recommendations = recommend_movies(movie_title, movies_df, movie_features, k)
    print(recommendations)


#question-2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
try:
    diabetes = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found.")
    exit()

# Handle missing data (Replace 0 values with NaN, then fill with mean)
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes[zero_columns] = diabetes[zero_columns].replace(0, np.nan)
diabetes[zero_columns] = diabetes[zero_columns].fillna(diabetes[zero_columns].mean())

# Split features and target
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']  # 0: Non-Diabetic, 1: Diabetic

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))