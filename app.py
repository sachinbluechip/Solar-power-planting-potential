import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set upload folder and allowed extensions for image uploads
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to apply morphological transformations to improve shadow detection
def improve_shadow_detection(shadow_mask):
    kernel = np.ones((5, 5), np.uint8)
    cleaned_shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    return cleaned_shadow_mask

# Function to segment different areas using k-means clustering
def segment_image(image, k=3):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image, labels.reshape((image.shape[0], image.shape[1]))

# Function to analyze the image for solar panel placement
def analyze_solar_potential(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segmented_image, labels = segment_image(image, k=3)
    avg_brightness = np.mean(gray_image)
    _, shadow_mask = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)
    cleaned_shadow_mask = improve_shadow_detection(shadow_mask)
   # Convert the shadow mask to a BGR image for coloring
    shadow_bgr = cv2.cvtColor(cleaned_shadow_mask, cv2.COLOR_GRAY2BGR)
    # Define a more subdued color for shadow areas (Dark Blue in BGR)
    subdued_color = (139, 0, 0)  # Dark Blue
    # Replace black areas (shadows) with the new color
    shadow_bgr[np.where((shadow_bgr == [0, 0, 0]).all(axis=2))] = subdued_color
    edges = cv2.Canny(gray_image, 100, 200)
    bright_area_mask = cv2.inRange(gray_image, 200, 255)
    solar_potential_score = np.mean(bright_area_mask) - np.mean(cleaned_shadow_mask)
    
    # Save results for display in the web interface
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results/')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save original image, segmented image, shadow detection, and edges
    original_image_path = os.path.join(output_folder, 'original_image.png')
    segmented_image_path = os.path.join(output_folder, 'segmented_image.png')
    shadow_image_path = os.path.join(output_folder, 'shadow_detection.png')
    edges_image_path = os.path.join(output_folder, 'edge_detection.png')
    
    cv2.imwrite(original_image_path, image)
    cv2.imwrite(segmented_image_path, segmented_image)
    cv2.imwrite(shadow_image_path, shadow_bgr)
    cv2.imwrite(edges_image_path, edges)
    
    # Return file paths relative to the static folder
    return avg_brightness, solar_potential_score, original_image_path, segmented_image_path, shadow_image_path, edges_image_path

def plot_solar_potential_score(scores, locations):
    plt.figure(figsize=(10,6))
    plt.bar(locations, scores, color=['green' if score > 150 else 'orange' for score in scores])
    plt.xlabel('Locations')
    plt.ylabel('Solar Potential Score')
    plt.title('Solar Potential Score by Location')
    plt.savefig(os.path.join('static/uploads/results/', 'solar_potential_score.png'))

def plot_brightness_shadow_heatmap(brightness_map):
    plt.figure(figsize=(8,6))
    plt.imshow(brightness_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Brightness/Shadow Intensity")
    plt.title('Brightness and Shadow Heatmap')
    plt.savefig(os.path.join('static/uploads/results/', 'brightness_shadow_heatmap.png'))

def plot_brightness_time_of_day(time_of_day, brightness_values):
    plt.figure(figsize=(10,6))
    plt.plot(time_of_day, brightness_values, marker='o')
    plt.xlabel('Time of Day')
    plt.ylabel('Average Brightness')
    plt.title('Average Brightness vs Time of Day')
    plt.savefig(os.path.join('static/uploads/results/', 'brightness_time_of_day.png'))

def plot_shadow_vs_bright_area(shadow_area, bright_area):
    labels = ['Shadow Area', 'Bright Area']
    sizes = [shadow_area, bright_area]
    colors = ['#1f77b4', '#ff7f0e']
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.title('Shadow vs Bright Area')
    plt.savefig(os.path.join('static/uploads/results/', 'shadow_vs_bright_area.png'))

# Example data after analyzing solar potential images
locations = ['Location 1', 'Location 2', 'Location 3']
solar_scores = [180, 140, 120]
brightness_map = np.random.rand(10, 10) * 255  # Example brightness/shadow map
time_of_day = np.linspace(6, 20, num=10)
brightness_values = np.random.rand(10) * 100
shadow_area = 40
bright_area = 60

# Generate the graphs based on the example data
plot_solar_potential_score(solar_scores, locations)
plot_brightness_shadow_heatmap(brightness_map)
plot_brightness_time_of_day(time_of_day, brightness_values)
plot_shadow_vs_bright_area(shadow_area, bright_area)



# Route to upload an image and analyze solar potential
@app.route("/")
def home():
    return render_template("hp.html")

@app.route("/login")
def login():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the uploaded image for solar potential
            avg_brightness, solar_potential_score, original_image_path, segmented_image_path, shadow_image_path, edges_image_path = analyze_solar_potential(filepath)
            
            # Set the caution message if the solar potential score exceeds 150
            
            caution_message = "A solar potential score greater than 150 indicates this is not an ideal location for solar panel installation."
            
            # Redirect to the results page
            return render_template("results.html", avg_brightness=avg_brightness, solar_potential_score=solar_potential_score,
                                   original_image=url_for('static', filename=os.path.join('uploads/results/', 'original_image.png')),
                                   segmented_image=url_for('static', filename=os.path.join('uploads/results/', 'segmented_image.png')),
                                   shadow_image=url_for('static', filename=os.path.join('uploads/results/', 'shadow_detection.png')),
                                   edges_image=url_for('static', filename=os.path.join('uploads/results/', 'edge_detection.png')),
                                   caution_message=caution_message)
    return render_template("upload.html")

# Route to display results
@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
