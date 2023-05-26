from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')
CORS(app)
# Load your RS model
model = load_model('Data/book/mae_best_model.h5')

# Load the data
ratings = pd.read_csv('Data/book/ratings.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data['user_id']

    # Get user's ratings
    user_ratings = ratings[ratings['user_id'] == user_id]

    # Find books the user hasn't rated yet
    unrated_books = ratings[~ratings['book_id'].isin(user_ratings['book_id'])]['book_id'].unique()

    # Create dataset for prediction
    user_id_arr = np.array([user_id for _ in range(len(unrated_books))])
    book_id_arr = np.array(unrated_books)
    predictions = model.predict([user_id_arr, book_id_arr])

    # Create a dataframe of predicted ratings
    predicted_ratings = pd.DataFrame({
        'user_id': user_id_arr,
        'book_id': book_id_arr,
        'predicted_rating': predictions.flatten(),
    })

    # Get the top 5 book recommendations
    recommended_books = predicted_ratings.sort_values(by='predicted_rating', ascending=False).head(5)

    return jsonify(recommended_books.to_dict())

if __name__ == '__main__':
    app.run(debug=True)

