from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import base64
from io import BytesIO
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'retinal'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'retinal_segmentation'

mysql = MySQL(app)

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)

def iou_score(y_true, y_pred, smooth=1.0):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
    return (intersection + smooth) / (union + smooth)

def specificity(y_true, y_pred):
    y_true_neg = 1 - y_true
    y_pred_neg = 1 - y_pred
    true_negatives = tf.reduce_sum(y_true_neg * y_pred_neg)
    possible_negatives = tf.reduce_sum(y_true_neg)
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
MODEL_PATH = 'fcn_retinal_vessels_final.keras'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MODEL_PATH'] = MODEL_PATH
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_once():
    global model
    if model is None:
        print(f"Loading model from {app.config['MODEL_PATH']}...")
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'iou_score': iou_score,
            'specificity': specificity,
            'sensitivity': sensitivity
        }
        model = load_model(app.config['MODEL_PATH'], custom_objects=custom_objects)
        print("Model loaded successfully!")
    return model


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", [email])
        user = cur.fetchone()
        if user:
            flash('Email already registered', 'error')
            cur.close()
            return render_template('register.html')
        hashed_password = generate_password_hash(password)
        cur.execute("INSERT INTO users(name, email, password, created_at) VALUES(%s, %s, %s, %s)",
                    (name, email, hashed_password, datetime.now()))
        mysql.connection.commit()
        cur.close()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password_candidate = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", [email])
        user = cur.fetchone()
        cur.close()
        if user:
            password = user[3]  # Assuming password is in the 4th column
            if check_password_hash(password, password_candidate):
                session['logged_in'] = True
                session['user_id'] = user[0]  # Assuming id is in the first column
                session['name'] = user[1]  # Assuming name is in the second column
                session['email'] = user[2]  # Assuming email is in the third column
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid password', 'error')
                return render_template('login.html')
        else:
            flash('Email not found', 'error')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'logged_in' in session:
        return render_template('dashboard.html')
    else:
        flash('Please login first', 'error')
        return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE id = %s", [session['user_id']])
        user = cur.fetchone()
        cur.close()
        if user:
            return render_template('profile.html', user=user)
        else:
            flash('User not found', 'error')
            return redirect(url_for('dashboard'))
    else:
        flash('Please login first', 'error')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate a unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(f"{unique_id}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image
        try:
            result = process_image(file_path)
            return jsonify(result)
        except Exception as e:
            error_message = str(e)
            print(f"Error processing image: {error_message}")
            return jsonify({'error': error_message})
    else:
        flash('File type not allowed')
        return redirect(request.url)


def preprocess_image(image_path, img_height=512, img_width=512):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Store original for display
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image
    img = cv2.resize(img, (img_width, img_height))

    # Convert to RGB if it's grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Normalize to [0,1]
    img = img / 255.0

    return img, original_img


def predict_segmentation(model, test_img):
    # Expand dimensions to create a batch of size 1
    test_img_batch = np.expand_dims(test_img, axis=0)

    # Get prediction
    prediction = model.predict(test_img_batch)[0]

    return prediction


def postprocess_prediction(pred_mask, threshold=0.5):
    # Threshold the prediction
    binary_mask = (pred_mask > threshold).astype(np.uint8)

    # Remove small connected components (noise)
    num_labels, labels = cv2.connectedComponents(binary_mask[:, :, 0])

    # Calculate area of each component
    for label in range(1, num_labels):
        component_size = np.sum(labels == label)
        # If component is too small, remove it
        if component_size < 50:  # Adjust this threshold as needed
            binary_mask[labels == label] = 0

    # Morphological operations to close small gaps
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert back to float and expand dimensions
    processed_mask = binary_mask.astype(np.float32)
    if len(processed_mask.shape) == 2:
        processed_mask = np.expand_dims(processed_mask, axis=-1)

    return processed_mask


def figure_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str


def process_image(image_path):
    # Load model
    model = load_model_once()

    # Process image
    processed_img, original_img = preprocess_image(image_path)

    # Get prediction
    pred_mask = predict_segmentation(model, processed_img)

    # Post-process prediction
    processed_mask = postprocess_prediction(pred_mask)

    # Calculate metrics
    dice = dice_coefficient(np.ones((1, 1)), np.ones((1, 1))).numpy()  # Just to get the correct format
    iou = iou_score(np.ones((1, 1)), np.ones((1, 1))).numpy()
    sens = sensitivity(np.ones((1, 1)), np.ones((1, 1))).numpy()
    spec = specificity(np.ones((1, 1)), np.ones((1, 1))).numpy()

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Segmentation overlay
    axes[1].imshow(processed_img)
    axes[1].imshow(pred_mask[:, :, 0], alpha=0.5, cmap='jet')
    axes[1].set_title('Segmented Vessels')
    axes[1].axis('off')

    plt.tight_layout()

    # Convert plot to base64 string
    img_str = figure_to_base64(fig)

    # Save raw prediction as image
    result_filename = os.path.basename(image_path).split('.')[0] + '_prediction.png'
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    cv2.imwrite(result_path, (pred_mask[:, :, 0] * 255).astype(np.uint8))

    # Calculate estimated metrics
    # Since we don't have ground truth, we'll use estimates based on vessel/non-vessel ratio
    vessel_ratio = np.mean(pred_mask > 0.5)
    estimated_metrics = {
        'Vessel Coverage': f"{vessel_ratio * 100:.2f}%",
        'Vessel Pixels': f"{int(np.sum(pred_mask > 0.5))}",
        'Total Pixels': f"{pred_mask.shape[0] * pred_mask.shape[1]}",
        'Prediction Confidence': f"{np.mean(np.abs(pred_mask - 0.5)) + 0.5:.2f}"
    }

    # Return results
    return {
        'image': img_str,
        'metrics': estimated_metrics,
        'result_path': url_for('static', filename=f'results/{result_filename}')
    }

if __name__ == '__main__':
    app.run(debug=True)