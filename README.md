# Machine_Learning_and_Data_Science
Repository for machine learning and data science projects.


Movie Analysis and Popularity-Based Filtering Project
Project Overview
This project utilizes Python and Pandas to analyze movie datasets. It includes scripts for loading data from CSV files, performing exploratory data analysis, and calculating weighted ratings for movies based on their popularity and critical acclaim.

Datasets
The project processes three main datasets:

movies.csv: Contains movie details like genres, links, titles, and budgets.
credits.csv: Includes cast and crew information for each movie.
ratings.csv: Provides user ratings for movies.
Installation
To run this project, you need Python and several libraries installed, including Pandas. You can set up your environment with the following steps:

bash
Copy code
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# It's recommended to set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install pandas
Usage
To run the script, navigate to the project directory and execute the Python script:

bash
Copy code
python movie_analysis.py
Here is a brief outline of what each part of the script does:

Load Data: Imports data from CSV files.
View Data: Displays the first few rows of the datasets to verify correct loading.
Calculate Weighted Ratings: Applies a formula to calculate a weighted rating for each movie based on votes and average ratings.
Example Analysis
Here's a snippet from the script that calculates the weighted ratings for movies:

python
Copy code
import pandas as pd

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Calculate a weighted rating for each movie
m = movies['vote_count'].quantile(0.9)
C = movies['vote_average'].mean()
def weighted_rating(x, m=m, C=C):
    V = x['vote_count']
    R = x['vote_average']
    return (V / (V + m) * R) + (m / (m + V) * C)

# Apply the weighted rating formula
movies['weighted_rating'] = movies.apply(weighted_rating, axis=1)



## NOTEBOOK 2 CONTENT-BASED FILTERING.ipynb:


Content-Based Movie Filtering Project
Project Overview
This project, hosted on Deepnote, applies content-based filtering to movie datasets to recommend movies similar to a given title based on their content descriptions. It utilizes Python, Pandas, and machine learning techniques (TF-IDF and cosine similarity) to analyze and predict preferences.

Features
Data Loading: Loads movie data from CSV files to perform analysis.
Text Analysis: Utilizes TF-IDF vectorization to transform text data from movie descriptions.
Similarity Calculation: Computes similarity scores between movies using cosine similarity to find matches based on content.
Movie Recommendations: Outputs movie recommendations based on content similarity.
Installation
This project is run on Deepnote, a collaborative data science platform. To get started:

Visit Deepnote and create an account if you don't have one.
Clone the project from the provided URL in the Deepnote environment.
The necessary libraries (Pandas, scikit-learn) are listed in the requirements.txt and can be installed directly within Deepnote.
How to Run
Once you have access to the project in Deepnote:

Open the movie_analysis.ipynb notebook.
Run the cells sequentially to load the data, perform TF-IDF vectorization, and calculate the similarity matrix.
Use the function similar_movies to get recommendations by entering the title of the movie and the number of recommendations desired.
Usage Example
python
Copy code
# Example of finding similar movies
similar_movies("Avatar", 5)
This function call would return five movies similar to "Avatar" based on the content descriptions provided in the dataset.


