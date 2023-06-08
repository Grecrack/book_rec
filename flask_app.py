from flask import Flask, request, jsonify, render_template, session
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, Flatten, Dot
from keras.models import Model
from sklearn.model_selection import train_test_split
from cryptography.fernet import Fernet
from multiprocessing import Value
import numpy as np
import pandas as pd
import pickle, requests, re, ast, threading

app = Flask(__name__)

with open('data/processed/user2user_encoded.pkl', 'rb') as f:
    user2user_encoded = pickle.load(f)

with open('data/processed/book2book_encoded.pkl', 'rb') as f:
    book2book_encoded = pickle.load(f)

with open('data/processed/book_id_to_name.pkl', 'rb') as f:
    book_id_to_name = pickle.load(f)

# Load the model
model = load_model('models/mae_best_model.h5')
ratings = pd.read_csv('data/ratings_n.csv')
books = pd.read_csv('data/books_n.csv')
users = pd.read_csv('data/users.csv') 
progress = Value('i', 0)  # Global variable to store progress

#API_key
encrypted_api_key = ('gAAAAABkf6RzFZqS9TqL5q3VvsfjB5zu0zl2utJ76pX0naJckdr7ElrrD_Fr2VJBe1xday0VQst21TPNnWbScepbRP__FEyiIPF1ytYMh0uWtLlILrmnoO9uHvpjIC62ctkw1pFFoYoM')
encrypted_search_engine_id = ('gAAAAABkf6Rz1kPvlkvE9bze6wSku0tEeogANyt8XlvjBMiUBJLzn9TaGiD0N6ps4BIMtlKPuv_rlrjnb2DmeFoDzZbNacZ4yAnou5ixE85nO53-LdosLVQ=')
with open('key','rb') as key_file:
    key = key_file.read()
key = ast.literal_eval(key.decode())

def decrypt_value(encrypted_value, key):
    f = Fernet(key)
    decrypted_value = f.decrypt(encrypted_value.encode())
    return decrypted_value.decode()

def update_rating(user_id, book_id, rating):
    global ratings
    # Check if the rating already exists
    if (ratings['book_id'].isin([int(book_id)]) & ratings['user_id'].isin([int(user_id)])).any():
        print('Rating already exists, update the existing rating')
        old_rating = ratings.loc[(ratings['book_id'] == int(book_id)) & (ratings['user_id'] == int(user_id)), 'rating'].values[0]
        ratings.loc[(ratings['book_id'] == int(book_id)) & (ratings['user_id'] == int(user_id)), 'rating'] = int(rating)
        books.loc[books['id'] == int(book_id), 'average_rating'] = ratings.loc[ratings['book_id'] == int(book_id), 'rating'].mean()
        books.loc[books['id'] == int(book_id), f'ratings_{rating}'] += 1
        books.loc[books['id'] == int(book_id), f'ratings_{old_rating}'] -= 1
    else:
        # Rating doesn't exist, add a new rating
        new_rating = pd.DataFrame({'book_id': [int(book_id)],'user_id': [int(user_id)],'rating': [int(rating)]})
        ratings = pd.concat([ratings, new_rating], ignore_index=True)
        # Update the ratings count and average rating in books.csv
        books.loc[books['id'] == int(book_id), 'ratings_count'] += 1
        books.loc[books['id'] == int(book_id), 'average_rating'] = ratings.loc[ratings['book_id'] == int(book_id), 'rating'].mean()
        # Update the specific ratings column in books.csv
        books.loc[books['id'] == int(book_id), f'ratings_{rating}'] += 1

    # Save the updated ratings DataFrame back to ratings.csv
    ratings.to_csv('data/ratings_n.csv', index=False)
    books.to_csv('data/books_n.csv', index=False)
    
    return 

def generate_new_user_id():
    # Generate a new user ID
    new_user_id = max(ratings['user_id']) + 1  # Assuming 'ratings' is your ratings dataset
    return new_user_id

def get_unrated_books(user_id):
    # Retrieve the unrated books for the given user ID
    rated_books = ratings[ratings['user_id'] == user_id]['book_id']
    all_book_ids = books['id']
    unrated_books = all_book_ids[~all_book_ids.isin(rated_books)]
    return list(unrated_books)

def sort_books_by_rating(books):
    # Sort the books based on average ratings from other users
    book_ratings = ratings.groupby('book_id')['rating'].mean()
    sorted_books = book_ratings.loc[books].sort_values(ascending=False)
    return sorted_books.index.tolist()

def recommend_books(user_id, num_books=5):
    # Encoding the user id if it exists in the mapping
    if user_id in user2user_encoded:
        user_encoded = user2user_encoded[user_id]
    else:
        # Handle the case where the user ID is not present in the mapping
        # You can return an empty list or provide a default recommendation in such cases
        return []

    # Getting the book ids in the encoding order
    book_ids = list(book2book_encoded.keys())
    book_ids = np.array(book_ids) - 1
    # Repeating the user id to match the shape of book ids
    user_array = np.array([user_encoded for _ in range(len(book_ids))])

    # Making the prediction
    pred_ratings = model.predict([user_array, np.array(book_ids)])

    # Getting the indices of the top num_books ratings
    top_indices = pred_ratings.flatten().argsort()[-num_books:][::-1]

    # Returning the corresponding book names
    recommended_books = [book_id_to_name[book_ids[i] + 1] for i in top_indices]
    return recommended_books

def transform_to_search_engine_friendly(title):
    # Remove special characters and spaces
    title = re.sub(r'[^\w\s-]', '', title)

    # Replace spaces with hyphens
    title = re.sub(r'\s', '-', title)

    # Convert to lowercase
    title = title.lower()

    return title

def give_rating(user_id):
    unrated_books = books[~books['id'].isin(ratings[ratings['user_id'] == int(user_id)]['book_id'])]
    sorted_books = unrated_books.sort_values(by='ratings_count', ascending=False)
    top_10_percent = sorted_books.head(int(len(sorted_books) * 0.1))
    book_id = top_10_percent.sample()['id'].values[0]
    book_title = books.loc[books['id'] == book_id, 'title'].values[0]
    image_url = books.loc[books['id'] == book_id, 'image_url'].values[0]
    return render_template('rate_book.html', user_id=user_id, book_id=book_id, book_title=book_title, image_url=image_url)

def retrain_model(ratings, user2user_encoded, book2book_encoded):
    ratings['user'] = ratings['user_id'].map(user2user_encoded)
    ratings['book'] = ratings['book_id'].map(book2book_encoded)
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    num_users = len(user2user_encoded)
    num_books = len(book2book_encoded)
    embedding_size = 10

    user_input = Input(shape=[1])
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    user_vec = Flatten()(user_embedding)

    book_input = Input(shape=[1])
    book_embedding = Embedding(num_books, embedding_size)(book_input)
    book_vec = Flatten()(book_embedding)

    product = Dot(axes=1)([book_vec, user_vec])

    model = Model(inputs=[user_input, book_input], outputs=product)
    mae_checkpoint_path = '../Data/mae_best_model.h5'

    # Define a callback for model checkpointing
    mae_checkpoint = ModelCheckpoint(mae_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    model.compile(loss=MeanAbsoluteError(), optimizer=Adam())
    print('loss function=MeanAbsoluteError()')
    print('optimizer=Adam()')
    print('batch_size=8')

    history = model.fit(x=[train.user.values, train.book.values], y=train.rating.values,
                        batch_size=64, epochs=1, verbose=1,
                        validation_data=([test.user.values, test.book.values], test.rating.values),
                        callbacks=[mae_checkpoint])

    print("Model update completed.")

def create_user():
    # Load the users.csv file
    users = pd.read_csv('data/users.csv')
    
    # Get the last user ID
    last_user_id = users['user_id'].max()
    
    # Generate a new user ID by incrementing the last user ID
    new_user_id = last_user_id + 1
    
    # Create a new user DataFrame
    new_user = pd.DataFrame({'user_id': [new_user_id]})
    
    # Concatenate the new user DataFrame with the existing users DataFrame
    users = pd.concat([users, new_user], ignore_index=True)
    
    # Save the updated users DataFrame to the users.csv file
    users.to_csv('data/users.csv', index=False)
    
    # Return the new user ID as a response to the user
    return new_user_id



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/new_user', methods=['GET'])
def new_user():
    return f"New user created with ID: {create_user()}"

@app.route('/train_user', methods=['GET', 'POST'])
def train_user():
    user_id = request.form.get('user_id')
    book_id = request.form.get('book_id')
    rating = request.form.get('rating')
    if user_id and rating:
        update_rating(user_id,book_id,rating)
        return give_rating(user_id)

    elif user_id:
        return give_rating(user_id)
    
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user ID from the form
    user_id = int(request.form.get('user_id'))

    # Get the recommended books for the user
    recommended_books = recommend_books(user_id)

    # Display the recommended books
    return render_template('recommended_books.html', books=recommended_books)

from threading import Thread

@app.route('/update_model', methods=['POST'])
def update_model():
    # get your data and encoding maps
    ratings=pd.read_csv('data/ratings_n.csv')
    user_ids = ratings['user_id'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    book_ids = ratings['book_id'].unique().tolist()
    book2book_encoded = {x: i for i, x in enumerate(book_ids)}
    Thread(target=retrain_model, args=(ratings, user2user_encoded, book2book_encoded)).start()

    return 'Model update started', 202





if __name__ == "__main__":
    app.run(debug=True)