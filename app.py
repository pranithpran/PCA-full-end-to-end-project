from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['COMPRESSED_FOLDER'] = 'compressed/'

# Ensure the folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['COMPRESSED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    image = request.files['image']
    confidence = request.form.get('confidence')

    if image.filename == '':
        return "No file selected", 400

    if image:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Convert confidence to float
        confidence_value = float(confidence)
        # Compress the image using PCA
        compressed_image_path = os.path.join(app.config['COMPRESSED_FOLDER'], 'compressed_image.jpg')
        reduce_image(image_path, compressed_image_path, confidence_value)

        # Redirect to display compressed image and download option
        return '', 204  # Return no content, JS will handle showing the download button
    
    return "Upload failed", 500

def reduce_image(input_path, output_path, confidence):
    # Load and process the image
    image = io.imread(input_path)
    gray_image = color.rgb2gray(image)
    
    # Apply PCA to compress
    pca = PCA(n_components=confidence)
    transformed_image = pca.fit_transform(gray_image)
    reconstructed_image = pca.inverse_transform(transformed_image)

    # Normalize and save compressed image
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)
    io.imsave(output_path, compressed_image_uint8)

@app.route('/download_compressed')
def download_compressed():
    # Serve the compressed image for download
    return send_from_directory(app.config['COMPRESSED_FOLDER'], 'compressed_image.jpg', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5050)
