import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load data for Popularity-Based Recommendations
try:
    with open("popular.pkl", "rb") as file:
        popular_books_df = pickle.load(file)
        
    # Check critical columns exist
    required_columns = ['Book-Title', 'Book-Author', 'Image-URL-M', 'Average_Rating', 'Total_number_of_Ratings']
    for col in required_columns:
        if col not in popular_books_df.columns:
            st.error(f"Column '{col}' missing in popular_books_df. Check data consistency.")
except FileNotFoundError:
    st.error("Popularity model file not found. Please ensure 'popular.pkl' is in the same directory.")

# Load data for Collaborative Filtering
try:
    pt = pickle.load(open('pt.pkl', 'rb'))
    merged = pickle.load(open('merged.pkl', 'rb'))
    score = pickle.load(open('score.pkl', 'rb'))
except FileNotFoundError:
    st.error("Collaborative Filtering model files not found. Please ensure 'pt.pkl', 'merged.pkl', and 'score.pkl' are in the same directory.")

# Streamlit UI
st.sidebar.title("Recommendation System")
option = st.sidebar.radio("Select Option", ("Popularity-Based", "Collaborative Filtering"))
st.title("Book Recommendation System")

if option == "Popularity-Based":
    st.header("Most Popular Books")
    
    if 'popular_books_df' in locals() and isinstance(popular_books_df, pd.DataFrame):
        st.write("### Top 50 Popular Books")
        cols = st.columns(3)
        
        for i, (index, row) in enumerate(popular_books_df.iterrows()):
            with cols[i % 3]:
                try:
                    st.image(row['Image-URL-M'], width=200, caption=row['Book-Title'])
                    st.markdown(f"""
                    **{row['Book-Title']}**  
                    *Author*: {row['Book-Author']}  
                    *Avg Rating*: {row['Average_Rating']:.1f}  
                    *Total Ratings*: {row['Total_number_of_Ratings']}
                    """)
                except KeyError as e:
                    st.error(f"Missing data for book: {e}")
                    break

elif option == "Collaborative Filtering":
    st.header("Collaborative Filtering Recommendations")
    book_name = st.selectbox("Search for a book:", pt.index.to_list())
    
    if book_name:
        index = np.where(pt.index == book_name)[0][0]
        similar_items = sorted(enumerate(score[index]), key=lambda x: x[1], reverse=True)[1:10]
        
        recommendations = []
        for i in similar_items:
            book_title = pt.index[i[0]]
            book_data = merged[merged['Book-Title'] == book_title].iloc[0]
            recommendations.append({
                'title': book_title,
                'author': book_data['Book-Author'],
                'image_url': book_data['Image-URL-M'],
                'avg_rating': book_data['Average_Rating'],
                'total_ratings': book_data['Total_number_of_Ratings']
            })
        
        cols = st.columns(3)
        for i, rec in enumerate(recommendations):
            with cols[i % 3]:
                st.image(rec['image_url'], width=200)
                st.markdown(f"""
                **{rec['title']}**  
                *Author*: {rec['author']}  
                *Avg Rating*: {rec['avg_rating']:.1f}  
                *Total Ratings*: {rec['total_ratings']}
                """)


# Now, the collaborative filtering section has an autocomplete search bar! 🚀