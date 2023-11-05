import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import pandas as pd

st.title('Movie Recommendation Sample test Version')
df = pd.read_csv(r'C:\\Users\\bhush\\Downloads\\Coursework\\INFO I 501\\project\\tmdb_tfidf1.csv')

search_query = st.text_input("Search for a Movie:")

# search query
suggestions = df['title'].str.contains(search_query, case=False, na=False)

# select box for suggestions
if suggestions.any():
    selected_suggestion = st.selectbox("Select the version:", df['title'][suggestions].dropna())
    row_index = df[df['title'] == selected_suggestion].index[0]

    if selected_suggestion:
        st.write("Row Index in the dataframe:", row_index)  


st.write("Few Suggestions")

tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the text data 
tfidf_matrix = tfidf_vectorizer.fit_transform(df['overview'])

def main_func(row_index):
    
    cosine_sim = {}
    i = 1
    while i < 11:
        k = random.randint(1, 760000)
        cosine_sim1 = cosine_similarity(tfidf_matrix[row_index], tfidf_matrix[k])
        cosine_sim[df['title'][i]] = cosine_sim1
        i = i + 1
    
    # Sort the dictionary 
    sorted_data = dict(sorted(cosine_sim.items(), key=lambda item: item[1][0][0], reverse=True))
    
    return sorted_data

# Calling the main function 
if 'row_index' in locals():
    recommendations = main_func(row_index)
    
    final_dict = {}
    for key, value in recommendations.items():
        final_dict[key] = np.array2string(value, separator=', ')[1:-1]

    st.write(final_dict)
