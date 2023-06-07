from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from cryptography.fernet import Fernet
import numpy as np
import pandas as pd
import pickle, requests, re, ast
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
    # Encoding the user id
    user_encoded = user2user_encoded[user_id]

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
    search_title = transform_to_search_engine_friendly(book_title)
    search_term = f"{search_title}+cover"
    api_key = decrypt_value(encrypted_api_key, key)
    search_engine_id = decrypt_value(encrypted_search_engine_id, key)
    search_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={search_term}"
    response = requests.get(search_url)
    search_results = response.json()
    first_item = search_results["items"][0]
    image_url = first_item["pagemap"]["scraped"][0]["image_link"]
    return render_template('rate_book.html', user_id=user_id, book_id=book_id, book_title=book_title, image_url=image_url)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/new_user', methods=['GET'])
def new_user():
    # Create a new user ID and share it with the user
    new_user_id = generate_new_user_id()
    return render_template('index.html', message=f"New user ID created: {new_user_id}")

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

if __name__ == "__main__":
    app.run(debug=True)