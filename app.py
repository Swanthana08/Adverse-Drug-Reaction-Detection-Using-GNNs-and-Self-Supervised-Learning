from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
import joblib
import numpy as np
import json
from werkzeug.security import generate_password_hash, check_password_hash
import os
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import nbformat
import nbformat
from nbconvert import HTMLExporter
from flask import Flask, render_template, session, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    mailid = db.Column(db.String(100))
    location = db.Column(db.String(100))
    mobile = db.Column(db.String(20))

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    filename = db.Column(db.String(100))
    result = db.Column(db.String(100))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        if User.query.filter_by(username=username).first():
            return "User already exists! Please log in."

        user = User(
            username=username,
            password=generate_password_hash(request.form['password']),
            mailid=request.form['mailid'],
            location=request.form['location'],
            mobile=request.form['mobile']
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid username or password."
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

# Load the model
model = load_model('18/18/Source Code/extension_weights.hdf5', compile=False)

# Constants
INPUT_DIM = 300  # Set your model's expected input vector length
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

import pandas as pd
dataset = pd.read_csv("18/18/Source Code/Dataset/twosides_drugbank.csv")
# Convert drug smile string into training vector so graph nodes can be created by GNN
dataset.drop(['type'], axis=1, inplace=True)
data_rows = dataset.values
# Combine text features for TF-IDF
X_raw = [f"{row[0]} {row[1]} {row[2]} {row[3]}" for row in data_rows]

# Fit TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
tfidf_vectorizer.fit(X_raw)
INPUT_DIM = tfidf_vectorizer.transform([X_raw[0]]).shape[1]

def predict_side_effect(model, input_vector):
    # Reshape input to match model's expected input shape: (None, 1351, 1, 1)
    input_vector = np.expand_dims(input_vector, axis=-1)  # Add channel dimension
    input_vector = np.expand_dims(input_vector, axis=-1)  # Add a singleton dimension for height
    input_vector = np.expand_dims(input_vector, axis=0)  # Add batch dimension
    
    prediction = model.predict(input_vector)
    return int(np.argmax(prediction))  # Assuming classification output

def vectorize_input(raw_input):
    try:
        # Input format: drug1_id|drug2_id|smiles1|smiles2
        if len(raw_input) != 4:
            raise ValueError("Input must contain 4 elements: drug1_id|drug2_id|smiles1|smiles2")

        drug1_id, drug2_id, smiles1, smiles2 = raw_input
        input_str = f"{drug1_id} {drug2_id} {smiles1} {smiles2}"
        input_vec = tfidf_vectorizer.transform([input_str]).toarray()

        if input_vec.shape[1] != INPUT_DIM:
            raise ValueError(f"Vectorized input dimension is {input_vec.shape[1]}, expected {INPUT_DIM}")

        return input_vec[0].astype(np.float32)

    except Exception as e:
        raise ValueError(f"Vectorization error: {str(e)}")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract input data from the form
        d1 = request.form.get('d1')
        d2 = request.form.get('d2')
        smiles1 = request.form.get('smiles1')
        smiles2 = request.form.get('smiles2')

        # Ensure that all input fields are provided
        if not all([d1, d2, smiles1, smiles2]):
            return "Error: All fields (d1, d2, smiles1, smiles2) are required."

        # Prepare the input for vectorization
        input_data = [d1, d2, smiles1, smiles2]
        print(input_data, "#########################")  # For debugging

        try:
            # Vectorize the input data
            input_vector = vectorize_input(input_data)

            # Predict side effects based on the vectorized input
            prediction = predict_side_effect(model, input_vector)

            # Format the predicted result
            detected_class = f"DDI type {prediction}"

            # Optionally save the prediction to the database if the user is logged in
            if 'username' in session:
                new_prediction = Prediction(
                    username=session['username'],
                    result=detected_class
                )
                db.session.add(new_prediction)
                db.session.commit()

            # Return the prediction result on the 'result.html' page
            return render_template('result.html', result=detected_class)

        except ValueError as ve:
            # Handle vectorization errors
            return f"Vectorization error: {str(ve)}"

        except Exception as e:
            # Catch any other errors
            return f"Error: {str(e)}"

    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Add your image file names here
    image_filenames = ["class.png", "accuracy.png", "confusion.png", "corr.png"]
    
    # Get the username from the session
    username = session.get('username')

    # Convert the Jupyter notebook to HTML for viewing in the web page
    notebook_path = '18/18/Source Code/Notebook.ipynb'  # Adjust this path
    
    try:
        # Check if the notebook file exists
        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"Notebook file not found at {notebook_path}")

        # Open the notebook file and read it
        with open(notebook_path, 'r') as notebook_file:
            notebook_content = nbformat.read(notebook_file, as_version=4)

        # Convert the notebook to HTML using HTMLExporter
        html_exporter = HTMLExporter()
        html_content, resources = html_exporter.from_notebook_node(notebook_content)
        
        # Save the notebook content as HTML to a temporary file for rendering
        html_file_path = 'static/your_notebook.html'
        with open(html_file_path, 'w') as html_file:
            html_file.write(html_content)

    except Exception as e:
        # Log error and return an error message
        print(f"Error: {str(e)}")
        return f"Error: Unable to process the notebook. {str(e)}"

    return render_template('analysis.html', images=image_filenames, username=username)
@app.route('/user_details')
def user_details():
    if 'username' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return "User not found."

    return render_template('user_details.html', user=user)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hardcoded admin login
        if username == 'Admin' and password == 'Admin':
            session['username'] = 'Admin'
            return redirect(url_for('admin_dashboard'))

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid username or password."
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'username' in session and session['username'] == 'Admin':
        users = User.query.all()
        predictions = Prediction.query.all()
        return render_template('admin_dashboard.html', users=users, predictions=predictions)
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Create tables if not exist
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create the tables before running the app
    app.run(debug=False)
