from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__,template_folder='templates')

# Load the model
model = load_model('data/mae_best_model.h5')
ratings = pd.read_csv('data/ratings.csv')
book = pd.read_csv('data/books.csv')

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
    all_book_ids = books['book_id']
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
    return [book_id_to_name[book_ids[i]] for i in top_indices]




@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        action = request.form.get('action')

        if action == 'new_user':
            # Create a new user ID and share it with the user
            new_user_id = generate_new_user_id()
            return render_template('index.html', message=f"New user ID created: {new_user_id}")

        elif action == 'train_user':
            # Get the user ID from the form
            user_id = int(request.form.get('user_id'))
            
            # Request ratings for books the user hasn't rated yet
            unrated_books = get_unrated_books(user_id)
            
            # Sort the unrated books by the average rating from other users
            sorted_books = sort_books_by_rating(unrated_books)
            
            # Display the books for the user to rate
            return render_template('rate_books.html', user_id=user_id, books=sorted_books)

        elif action == 'recommend_books':
            # Get the user ID from the form
            user_id = int(request.form.get('user_id'))

            # Get the recommended books for the user
            recommended_books = recommend_books(user_id)

            # Display the recommended books
            return render_template('recommended_books.html', books=recommended_books)

    return render_template('index.html')

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