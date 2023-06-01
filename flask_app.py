from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)


# Load the model
model = load_model('data/mae_best_model.h5')
ratings = pd.read_csv('data/ratings.csv')
books = pd.read_csv('data/books.csv')

with open('data/processed/user2user_encoded.pkl', 'rb') as f:
    user2user_encoded = pickle.load(f)

with open('data/processed/book2book_encoded.pkl', 'rb') as f:
    book2book_encoded = pickle.load(f)

with open('data/processed/book_id_to_name.pkl', 'rb') as f:
    book_id_to_name = pickle.load(f)
    

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


user_id=0


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
    if request.method == 'POST':
        user_id = request.form.get('user_id')

        if user_id:
            # Get the books that the user hasn't rated yet
            unrated_books = books[~books['id'].isin(ratings[ratings['user_id'] == int(user_id)]['book_id'])]

            if len(unrated_books) > 0:
                # Sort the unrated books by rating in descending order
                sorted_books = unrated_books.sort_values(by='average_rating', ascending=False)

                # Get the highest-rated book
                book_id = sorted_books.iloc[0]['book_id']
                book_title = sorted_books.iloc[0]['title']

                return render_template('rate_book.html', user_id=user_id, book_id=book_id, book_title=book_title)
            else:
                return 'No unrated books found for the user.'

    # Handle GET request
    return render_template('index.html')
@app.route('/save_rating', methods=['POST'])
def save_rating():
    user_id = request.form.get('user_id')
    book_id = request.form.get('book_id')
    book_title = request.form.get('book_title')
    rating = request.form.get('rating')

    # Save the rating to the ratings.csv file or your desired storage

    return f"Rating {rating} saved for User {user_id} and Book {book_title}."






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





#@app.route('/', methods=['GET'])
#def home():
#    return render_template('index.html')

#@app.route('/recommend', methods=['POST'])
#def recommend():
#    user_id = int(request.form.get('user_id'))
#    recommended_books = recommend_books(user_id)
#    return render_template('index.html', books=recommended_books)

#if __name__ == "__main__":
#    app.run(debug=True)