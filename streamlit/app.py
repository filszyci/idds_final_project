# PAKIETY
import streamlit as st
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import mysql.connector
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from datetime import datetime
import warnings; warnings.simplefilter('ignore')

#1. KONFIGURACJA
# Połącz z bazą danych MySQL
conn_str = 'mysql+pymysql://streamlit_user:streamlit@localhost:3306/movies_transactions'
engine = create_engine(conn_str)
metadata = MetaData()
metadata.reflect(bind=engine)


users_unique = pd.read_sql('SELECT * FROM movies_dwh.users', con=engine)
max_id = users_unique['user_id'].max() + 1

# ORM
users = Table('users', metadata, 
              Column('user_id', Integer, primary_key=True, autoincrement=True),
              Column('user_name', String(50)),extend_existing=True)
           
#metadata.tables['users']
ratings = metadata.tables['ratings']

# Konfiguracja sesji
Session = sessionmaker(bind=engine)
session = Session()

# 2. FUNKCJE
# Definicja modelu User
Base = declarative_base()
# Zapis nowego użytkownika w bazie danych
class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String(50))

def add_user_(session, user_name):
    new_user = User(user_name=user_name)
    session.add(new_user)
    session.commit()
    return new_user

# Funkcja do dodawania nowego użytkownika
def add_user(user_name):
    new_user = users.insert().values(user_name=user_name)
    result = engine.execute(new_user)
    return result.inserted_primary_key[max_id]

def add_rating(user_id, movie_id, rating):
    timestamp = int(datetime.now().timestamp())
    new_rating = ratings.insert().values(user_id=user_id, movie_id=movie_id, rating=rating, timestamp=timestamp)
    engine.execute(new_rating)

# Obliczanie SVD
@st.cache_data
def predict_rating(movie_id):
    return svd.predict(selected_user, movie_id).est

#4. BAZOWY LAYOUT

selected_user = st.sidebar.selectbox('Select User', users_unique['user_id'].unique())

#3. WCZYTYWANIE DANYCH
# Pobieranie tabel i widoków
ratings_history = pd.read_sql('SELECT * FROM movies_dwh.v_ml_ratings', con=engine)

#movies_metadata
movies_metadata = pd.read_sql('SELECT * FROM movies_dwh.v_movies_metadata', con=engine)

#user-user
user_user = pd.read_sql('SELECT * FROM movies_dwh.ml_result_user_user WHERE user_id < 100', con=engine)

#movie-movie
movie_movie = pd.read_sql('SELECT * FROM movies_dwh.ml_result_movie_movie WHERE user_id < 100', con=engine)


    

#5. Transformacje danych
right_mm = movies_metadata[['movie_id_ratings', 'title', 'release_date']].rename(columns={'movie_id_ratings': 'movie_id'})
filtered_df = ratings_history[ratings_history['user_id'] == selected_user]
user_ratings = pd.merge(filtered_df, right_mm, how='left', on='movie_id').drop('movie_id', axis=1)
user_ratings_selected = user_ratings[['title', 'release_date', 'rating', 'rating_date']].rename(columns={
    'title': 'Title', 'release_date': 'Release Date', 'rating': 'Rating', 'rating_date': 'Rating Date'
})
user_ratings_selected['Rating'] = user_ratings_selected['Rating'].apply(lambda x: '{:.2f}'.format(x))

user_user_merged = pd.merge(user_user, right_mm, on='movie_id', how='left').rename(columns={
    'title': 'Title', 'release_date': 'Release Date', 'predicted_rating': 'Predicted Rating'
})
user_user_filtered = user_user_merged[user_user_merged['user_id'] == selected_user]
user_user_filtered = user_user_filtered[['Title', 'Release Date', 'Predicted Rating']]

movie_movie_renamed = movie_movie.rename(columns={
    'movie_id': 'best_movie', 'similar_movies': 'movie_id'})
movie_movie_merged = pd.merge(movie_movie_renamed, right_mm, on='movie_id', how='left').rename(columns={
    'title': 'Title', 'release_date': 'Release Date', 'predicted_rating': 'Predicted Rating'
})
movie_movie_filter = movie_movie_merged[movie_movie_merged['user_id'] == selected_user]
movie_movie_filtered = movie_movie_filter[['Title', 'Release Date', 'Predicted Rating']]
best_movie = pd.merge(movie_movie[movie_movie['user_id']==selected_user], right_mm, on='movie_id', how='left')['title'].unique()[0]

#uczenie maszynowe

ratings = pd.read_sql('SELECT * FROM movies_dwh.v_ml_ratings', con = engine)
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
trainset = data.build_full_trainset()
svd.fit(trainset)

users = ratings['user_id'].unique()
movies = set(ratings['movie_id'].unique())
movies_predictions_df = pd.DataFrame(columns=['movie_id', 'user_id', 'predicted_rating'])

movies_rated = set(ratings[ratings['user_id'] == selected_user]['movie_id'].to_list())
movies_unrated = list(movies - movies_rated)
movies_predictions = pd.DataFrame(movies_unrated, columns=['movie_id'])
movies_predictions['user_id'] = selected_user
movies_predictions['predicted_rating'] = movies_predictions['movie_id'].apply(predict_rating)
movies_predictions_df = pd.concat([movies_predictions_df, movies_predictions], axis=0)
top_predictions = movies_predictions_df.loc[movies_predictions.nlargest(10, 'predicted_rating').index]
movies_predictions_merged = pd.merge(top_predictions, right_mm, on='movie_id', how='left')
movies_predictions_selected = movies_predictions_merged[['title', 'release_date', 'predicted_rating']].rename(columns={
    'title': 'Title', 'release_date': 'Release Date', 'predicted_rating': 'Predicted Rating'}).sort_values(by='Predicted Rating', ascending=False)
movies_predictions_selected['Predicted Rating'] = movies_predictions_selected['Predicted Rating'].apply(lambda x: '{:.2f}'.format(x))

#----------------------------------------------------------------

# LAYOUT
st.title(f"Rekomendacje dla użytkownika o id={selected_user}")

col1, col2 = st.columns(2)

st.sidebar.title('Dodawanie Nowego Użytkownika')

user_name = st.sidebar.text_input('Nazwa Użytkownika')

if st.sidebar.button('Dodaj Użytkownika'):
    if user_name:
        new_user = add_user_(session, user_name)
        st.success(f'Nowy użytkownik dodany: {new_user.user_name}')
    else:
        st.error('Proszę wprowadzić nazwę użytkownika.')

with col1:
    st.markdown(f"Filmy proponowane na podstawie ocen innych użytkowników")
    st.dataframe(movies_predictions_selected, hide_index=True)

    st.markdown(f"Filmy proponowane na podstawie filtrowania user-user")
    st.dataframe(user_user_filtered, hide_index=True)

    st.markdown(f"Ponieważ użytkownik obejrzał {best_movie}")
    st.dataframe(movie_movie_filtered, hide_index=True)
    

with col2:
    st.markdown(f"Historia dotychczasowych ocen bieżącego użytkownika")
    st.dataframe(user_ratings_selected, hide_index=True)
    
    if 'user_id' not in st.session_state:
        user_name = st.text_input("Enter your name")
        if user_name:
            user_id = add_user(user_name)
            st.session_state['user_id'] = user_id
    else:
        user_id = st.session_state['user_id']
        user_name_query = select([users.c.user_name]).where(users.c.user_id == user_id)
        user_name = engine.execute(user_name_query).fetchone()[0]
        st.write(f"User ID: {user_id} (Name: {user_name})")

    #st.write(f'User ID: {user_id}')
    
    #movie_id = st.number_input("Enter Movie ID", min_value=1, step=1)
    movie_title = st.selectbox("Select Movie", right_mm['title'])
    selected_movie_id = right_mm[right_mm['title'] == movie_title]['movie_id'].values[0]

    rating = st.slider("Rate the Movie", min_value=0.0, max_value=5.0, step=0.5)

    if st.button("Submit Rating"):
        add_rating(selected_user, selected_movie_id, rating)
        st.success("Rating submitted successfully!")