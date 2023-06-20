from flask import Flask, request, render_template, Response, url_for, redirect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from keras.models import Model,load_model
from keras.optimizers import RMSprop
from keras.losses import LogCosh
from keras.layers import Input, Embedding, Flatten,Dot
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os,  math

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')
training_completed = False
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
class User(UserMixin):
    def __init__(self, id):
        self.id = id

    @classmethod
    def get_user(cls, user_id):
        users = pd.read_csv('data/users.csv')
        if str(user_id) in users['user_id'].astype(str).values: 
            return cls(user_id)
        return None

@login_manager.user_loader
def load_user(user_id):
    users = pd.read_csv('data/users.csv')

    if str(user_id) in users['user_id'].astype(str).values: 
        return User(user_id)
    else:
        return None

def create_model(model,num_users,num_books):
    embedding_dim = 10
    user_embedding_layer = model.get_layer('user_embedding')
    user_embedding_config = user_embedding_layer.get_config()
    user_id_max = user_embedding_config['input_dim']
    print(user_id_max)
      
    old_weights = model.get_layer('user_embedding').get_weights()[0]
    new_weights = np.zeros((num_users , embedding_dim))
    new_weights[:user_id_max,0 :] = old_weights
    new_user_embedding = Embedding(num_users, embedding_dim, input_length=1)

    new_user_embedding.build((None,)) 
    new_user_embedding.set_weights([new_weights])
    new_user_embedding._name = "user_embedding"

    user_input = Input(shape=(1,), name="user_input")
    user_embedding = new_user_embedding(user_input)
    user_vec = Flatten()(user_embedding)

    book_input = Input(shape=[1], name="book_input")
    book_embedding = Embedding(num_books, embedding_dim, name="book_embedding")(book_input)
    book_vec = Flatten()(book_embedding)

    product = Dot(axes=1)([book_vec, user_vec])

    new_model = Model(inputs=[user_input, book_input], outputs=product)
    for layer_index, layer in enumerate(model.layers):
        if layer_index >= 2 and layer.name != 'user_embedding':
            new_model.layers[layer_index].set_weights(layer.get_weights())
    new_model.compile(loss=model.loss, optimizer=model.optimizer)
    return new_model

def retrain_model():
    ratings=pd.read_csv('Data/ratings.csv')
    optimizer = RMSprop()
    loss_function = LogCosh()
    batch_size = 16
    model_path = ('models/model_Adam_Huber_batch16.tf')
    model = load_model(model_path)
    user2user_encoded = {x: i for i, x in enumerate(ratings['user_id'].unique())}
    book2book_encoded = {x: i for i, x in enumerate(ratings['book_id'].unique())}
    book_ids = ratings['book_id'].unique().tolist()
    book_encoded2book = {i: x for i, x in enumerate(book_ids)}
    num_users = len(user2user_encoded) 
    num_books = len(book_encoded2book)
    ratings['user'] = ratings['user_id'].map(user2user_encoded)
    ratings['book'] = ratings['book_id'].map(book2book_encoded)
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    new_model = create_model(model,num_users,num_books)
    for layer_index, layer in enumerate(model.layers):
        if layer_index >= 2 and layer.name != 'user_embedding':
            new_model.layers[layer_index].set_weights(layer.get_weights())
    new_model.compile(loss=loss_function, optimizer=optimizer)
    new_model.fit(
        x=[train['user'].values, train['book'].values],y=train['rating'].values,
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        validation_data=([test['user'].values, test['book'].values], test['rating'].values)) 
    new_model.save(model_path)
    return

def recommend_books(user_id, num_books=5):

            model = load_model('models/model_Adam_Huber_batch16.tf')
            ratings = pd.read_csv('data/ratings.csv')
            books = pd.read_csv('data/books.csv')
            book_id_to_name = pd.Series(books.title.values, index = books.index).to_dict()
            user_ids = ratings['user_id'].unique().tolist()
            user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            book_ids = ratings['book_id'].unique().tolist()
            book2book_encoded = {x: i for i, x in enumerate(book_ids)}

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
            recommended_books = []
            for i in top_indices:
                book_id = book_ids[i] + 1
                book_title = book_id_to_name[book_id]
                book_image_url = books.loc[books['title'] == book_title, 'image_url'].values[0]
                amazon_link = books.loc[books['title'] == book_title, 'amazon_link'].values[0]
                recommended_books.append({
                    "title": book_title,
                    "image": book_image_url,
                    "amazon_link": amazon_link
                })
            return recommended_books

def update_rating(user_id, book_id, rating):
    ratings = pd.read_csv('data/ratings.csv')
    books = pd.read_csv('data/books.csv')
    users = pd.read_csv('data/users.csv')
    # Check if the rating already exists
    print('user', user_id, 'rated', book_id, 'with', rating)
    if (ratings['book_id'].isin([int(book_id)]) & ratings['user_id'].isin([int(user_id)])).any():
        print('Rating already exists, updating the existing rating')
        old_rating = ratings.loc[
            (ratings['book_id'] == int(book_id)) & (ratings['user_id'] == int(user_id)), 'rating'].values[0]
        ratings.loc[(ratings['book_id'] == int(book_id)) & (ratings['user_id'] == int(user_id)), 'rating'] = int(
            rating)
        books.loc[books['id'] == int(book_id), 'average_rating'] = ratings.loc[
            ratings['book_id'] == int(book_id), 'rating'].mean()
        books.loc[books['id'] == int(book_id), f'ratings_{rating}'] += 1
        books.loc[books['id'] == int(book_id), f'ratings_{old_rating}'] -= 1

        print('rating', old_rating, 'replaced with', rating)
    else:
        # Rating doesn't exist, add a new rating
        new_rating = pd.DataFrame({'book_id': [int(book_id)], 'user_id': [int(user_id)], 'rating': [int(rating)]})
        ratings = pd.concat([ratings, new_rating], ignore_index=True)
        # Update the ratings count and average rating in books.csv
        books.loc[books['id'] == int(book_id), 'ratings_count'] += 1
        books.loc[books['id'] == int(book_id), 'average_rating'] = ratings.loc[
            ratings['book_id'] == int(book_id), 'rating'].mean()
        # Update the specific ratings column in books.csv
        books.loc[books['id'] == int(book_id), f'ratings_{rating}'] += 1
        users.loc[ratings['user_id'] == int(user_id), f'rating_count'] += 1
        print('rating', new_rating, 'added to', book_id, 'for',user_id)


    # Save the updated ratings DataFrame back to ratings.csv
    users.loc[users['user_id'] == int(user_id), 'new_data'] = True
    ratings.to_csv('data/ratings.csv', index=False)
    books.to_csv('data/books.csv', index=False)
    users.to_csv('data/users.csv', index=False)
    print('update rating.csv and book.csv')
    return give_rating(user_id)

def get_unrated_books(user_id):
    ratings= pd.read_csv('data/ratings.csv')
    books = pd.read_csv('data/books.csv')
    rated_books = ratings[ratings['user_id'] == user_id]['book_id']
    all_book_ids = books['id']
    unrated_books = all_book_ids[~all_book_ids.isin(rated_books)]
    return list(unrated_books)

def sort_books_by_rating(books):
    ratings= pd.read_csv('data/ratings.csv')
    book_ratings = ratings.groupby('book_id')['rating'].mean()
    sorted_books = book_ratings.loc[books].sort_values(ascending=False)
    return sorted_books.index.tolist()

def progress_updates():
    def generate():
        progress = 0
        while not training_completed:
            yield f"data: {progress}\n\n"
            time.sleep(0.1)
        progress = 100
        yield f"data: {progress}\n\n"
    return Response(generate(), mimetype='text/event-stream')

def give_rating(user_id):
    ratings= pd.read_csv('data/ratings.csv')
    books = pd.read_csv('data/books.csv')
    unrated_books = books[~books['id'].isin(ratings[ratings['user_id'] == int(user_id)]['book_id'])]
    sorted_books = unrated_books.sort_values(by='ratings_count', ascending=False)
    top_10_percent = sorted_books.head(int(len(sorted_books) * 0.1))
    book_id = top_10_percent.sample()['id'].values[0]
    book_title = books.loc[books['id'] == book_id, 'title'].values[0]
    image_url = books.loc[books['id'] == book_id, 'image_url'].values[0]
    print(book_id, book_title, image_url)
    return render_template('rate_book.html', user_id=user_id, book_id=book_id, book_title=book_title,
                           image_url=image_url)
               
def create_user():
    ratings=pd.read_csv('data/ratings.csv')
    books=pd.read_csv('data/books.csv')
    users = pd.read_csv('data/users.csv')
    filtered_books5 = books[books['average_rating'] >= 4]
    filtered_books4 = books[(books['average_rating'] >= 3.0) & (books['average_rating'] < 4)]
    filtered_books3= books[(books['average_rating'] >= 2.0) & (books['average_rating'] < 3.0)]
    filtered_books2= books[(books['average_rating'] >= 1.0) & (books['average_rating'] < 2.0)]
    filtered_books1= books[(books['average_rating'] >= 0) & (books['average_rating'] < 1)]
    top_books_5 = filtered_books5[['id', 'ratings_5']]
    top_books_4 = filtered_books4[['id', 'ratings_4']]
    top_books_3 = filtered_books3[['id', 'ratings_3']]
    top_books_2 = filtered_books2[['id', 'ratings_2']]
    top_books_1 = filtered_books1[['id', 'ratings_1']]
    filtered_top_books_5 = top_books_5.nlargest(20, 'ratings_5').tail(10)
    filtered_top_books_4 = top_books_4.nlargest(20, 'ratings_4').tail(10)
    filtered_top_books_3 = top_books_3.nlargest(20, 'ratings_3').tail(10)
    filtered_top_books_2 = top_books_2.nlargest(20, 'ratings_2').tail(10)
    filtered_top_books_1 = top_books_1.nlargest(20, 'ratings_1').tail(10)
    # Get the last user ID from the users.csv file
    last_user_id = users['user_id'].max()

    # Check if the last user ID is present in the ratings.csv file
    if not ratings[(ratings['user_id'] == last_user_id) & (ratings['cold_start'] == False)].empty:
        new_user_id = last_user_id
    else:
        # Generate a new user ID by incrementing the last user ID
        new_user_id = last_user_id + 1

    # Create a new user DataFrame
    new_user = pd.DataFrame({'user_id': [new_user_id]})

    # Concatenate the new user DataFrame with the existing users DataFrame
    users = pd.concat([users, new_user], ignore_index=True)

    users.loc[users['user_id'] == int(new_user_id), 'rating_count'] = 0

    # Save the updated users DataFrame to the users.csv file
    users.to_csv('data/users.csv', index=False)

    # Return the new user ID as a response to the user
    
    # Set the rating to 1-5
    filtered_top_books_5['rating'] = 5
    filtered_top_books_4['rating'] = 4
    filtered_top_books_3['rating'] = 3
    filtered_top_books_2['rating'] = 2
    filtered_top_books_1['rating'] = 1

    # Add a column for user_id

    filtered_top_books_5['user_id'] = user_id
    filtered_top_books_4['user_id'] = user_id
    filtered_top_books_3['user_id'] = user_id
    filtered_top_books_2['user_id'] = user_id
    filtered_top_books_1['user_id'] = user_id

    # Concatenate all DataFrames into one
    coldstart_df = pd.concat([filtered_top_books_5[['id', 'user_id', 'rating']],
                            filtered_top_books_4[['id', 'user_id', 'rating']],
                            filtered_top_books_3[['id', 'user_id', 'rating']],
                            filtered_top_books_2[['id', 'user_id', 'rating']],
                            filtered_top_books_1[['id', 'user_id', 'rating']]])
    coldstart_df = coldstart_df.rename(columns={"id":"book_id"})
    coldstart_df['cold_start'] = True


    ratings = pd.concat([ratings, coldstart_df], ignore_index=True)
    return(new_user_id)

def get_user_ratings(user_id):
    ratings = pd.read_csv('data/ratings.csv')
    users = pd.read_csv('data/users.csv')
    books = pd.read_csv('data/books.csv')

    user_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['cold_start'] == False)]
    rating_count = users[users['user_id'] == user_id]['rating_count'].values[0]
    
    if not user_ratings.empty:
        books = books[['id', 'title']]
        user_ratings = user_ratings.merge(books, left_on='book_id', right_on='id')
        user_ratings = user_ratings.drop(columns=['user_id'])

    return user_ratings, int(rating_count)

def load_user(user_id):
    return User(user_id)

#----------------------------------------------------------------------------------------#

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['POST','GET'])
def login():
    try:
        user_id = request.form['user_id']
        user = User.get_user(user_id)

        if user is not None:
            login_user(user)
            return redirect(url_for('main'))
        else:
            return 'Invalid user_id'
    except KeyError:
        if current_user.is_authenticated:
            user_id = current_user.id
            return redirect(url_for('main'))
        else:
            return 'No user_id provided and no user logged in'

@app.route('/main', methods=['POST','GET'])
@login_required
def main():
    user_id = int(current_user.id)
    if user_id is not None:
        return render_template('user.html', user_id=user_id)
    return "No user id provided", 400

@app.route('/new_user', methods=['GET'])
def new_user():
    user_id = create_user()
    user = User.get_user(user_id)    
    if user:
        login_user(user)  # Log in the user
        
        # Redirect to the desired route, e.g., the user's profile page
        return render_template('user.html', user_id=user_id)
    
    # Handle the case if the user is not found
    return render_template('home.html', error="Failed to create new user")


@app.route('/train_user', methods=['POST','GET'])
@login_required
def train_user():
    users = pd.read_csv('data/users.csv')
    user_id = int(current_user.id)
    book_id = None
    rating = None
    if 'book_id' in request.form:
        book_id = int(request.form.get('book_id'))
    if 'rating' in request.form:
        rating = int(request.form.get('rating'))
    if (book_id!=None) & (rating!=None):
        return update_rating(user_id, book_id, rating)
    if str(user_id) in users['user_id'].astype(str).values:        
        return give_rating(user_id)
    else:
        return render_template('user.html', error=f"User ID {user_id} not found.")

@app.route('/recommend', methods=['POST','GET'])
@login_required
def recommend():
            users = pd.read_csv('data/users.csv')
        #try:
            user_id = int(current_user.id)
            if int(user_id) in users['user_id'].astype(int).values:
                recommended_books = recommend_books(user_id)
                return render_template('recommended_books.html', books=recommended_books)
            else:
                return render_template('user.html', error=f"User ID {user_id} not found.")
        #except:
            #return render_template('user.html', error=f"User ID {user_id} not found in Training Data")

@app.route('/update_model', methods=['POST'])
@login_required
def update_model():
    global training_completed
    training_completed = False
    ratings = pd.read_csv('data/ratings.csv')
    user_ids = ratings['user_id'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    book_ids = ratings['book_id'].unique().tolist()
    book2book_encoded = {x: i for i, x in enumerate(book_ids)}
    retrain_model()
    training_completed = True

    return 'Model update completed', 202

@app.route('/view_profile', methods=['GET','POST'])
@login_required
def view_profile():
    user_id = int(current_user.id)
    user_ratings, rating_count = get_user_ratings(user_id)

    page = request.args.get('page', default=1, type=int)
    per_page = 5
    total_pages = math.ceil(len(user_ratings) / per_page)
    paginated_ratings = user_ratings[(page-1)*per_page: page*per_page]

    return render_template('profile.html', user_id=user_id, ratings=paginated_ratings.to_dict('records'),
                           rating_count=rating_count, page=page, total_pages=total_pages)




@app.route('/logout', methods=['POST','GET'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True,port=5001)