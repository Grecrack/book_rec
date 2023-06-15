# Book Recommendation System

This repository contains the code for a Book Recommendation System. The system allows users to rate books, receive personalized book recommendations, and view their profile and ratings.

## Features

- Users can create a new account and receive a unique user ID.
- Users can rate books they have read.
- The system recommends books to users based on their ratings and the ratings of other users.
- Users can view their profile and see their ratings.
- The system periodically updates the recommendation model with new user ratings.

## Requirements

To run the code in this repository, you need the following:

- Python 3.x
- Flask
- Keras
- pandas
- numpy
- scikit-learn

## Getting Started

1. Clone the repository:

   ```bash
   git clone [(https://github.com/Grecrack/book_rec)[-url>]](https://github.com/Grecrack/book_rec.git)](https://github.com/Grecrack/book_rec.git)
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Open your browser and go to `http://localhost:5000` to access the Book Recommendation System.

## Repository Structure

- `app.py`: The main Flask application file that handles the routes and logic for the Book Recommendation System.
- `data/`: Directory containing the CSV files for ratings, books, and users.
- `models/`: Directory containing the pre-trained recommendation models.
- `templates/`: Directory containing the HTML templates for the user interface.

## Model Training

The recommendation models are trained using the `train_model.py` script. The script reads the ratings data from the `data/ratings.csv` file, creates embedding layers for users and books, trains the model, and saves the best model based on validation loss.

To train the models, run the following command:

```bash
python flask_app.py
```

## Contributors

- Dimitris Gkrekas https://github.com/Grecrack

Feel free to contribute to this project by submitting pull requests or creating issues.

## License

This project is licensed under the [MIT License](LICENSE).
```

