from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = load_model('mae_best_model.h5')

with open('user2user_encoded.pkl', 'rb') as f:
    user2user_encoded = pickle.load(f)

with open('book2book_encoded.pkl', 'rb') as f:
    book2book_encoded = pickle.load(f)

with open('book_id_to_name.pkl', 'rb') as f:
    book_id_to_name = pickle.load(f)


def recommend_books(user_id, num_books=5):
    # Encoding the user id
    user_encoded = user2user_encoded[user_id]

    # Getting the book ids in the encoding order
    book_ids = list(book2book_encoded.keys())

    # Repeating the user id to match the shape of book ids
    user_array = np.array([user_encoded for _ in range(len(book_ids))])

    # Making the prediction
    pred_ratings = model.predict([user_array, np.array(book_ids)])

    # Getting the indices of the top num_books ratings
    top_indices = pred_ratings.flatten().argsort()[-num_books:][::-1]

    # Returning the corresponding book names
    return [book_id_to_name[book_ids[i]] for i in top_indices]


@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    recommended_books = recommend_books(user_id)

    return jsonify(recommended_books)


if __name__ == "__main__":
    app.run(debug=True)

