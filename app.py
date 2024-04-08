from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np
from image_processing import process_image_notebook
from werkzeug.utils import secure_filename
import cv2
import base64
import io

app = Flask(__name__, static_folder='static')

def process_image(image_path):
    original_img, output_img = process_image_notebook(image_path)

    # Convert the output image to PIL format
    output_image = Image.fromarray(output_img)

    # Compress the original image
    compressed_original_image = compress_image(original_img)

    return compressed_original_image, output_image

def compress_image(image, max_size=1000):
    """Compress a NumPy array image to a maximum size while maintaining its aspect ratio."""
    if len(image.shape) == 3:
        # Color image
        height, width, _ = image.shape
    else:
        # Grayscale image
        height, width = image.shape

    # Calculate the scaling factor to fit the image within the maximum size
    aspect_ratio = min(max_size / width, max_size / height)
    new_width = int(width * aspect_ratio)
    new_height = int(height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Encode the resized image as JPEG
    _, encoded_image = cv2.imencode('.jpg', resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    return encoded_image

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Process the image
            compressed_original_image, output_image = process_image(file_path)

            # Convert the compressed original image to a base64-encoded string
            compressed_original_img_base64 = base64.b64encode(compressed_original_image).decode('utf-8')
            compressed_original_img_url = f"data:image/jpeg;base64,{compressed_original_img_base64}"

            # Get the original filename and extension
            original_filename, extension = os.path.splitext(filename)

            # Construct the modified filename
            modified_filename = f"{original_filename}_modified{extension}"

            # Save the output image with the modified filename
            output_path = os.path.join('static', modified_filename)
            output_image.save(output_path)

            # Construct the output image URL
            output_image_url = url_for('static', filename=modified_filename)

            # Render the template and pass the output_image and compressed_original_img_url variables
            return render_template('index.html', output_image=output_image_url, original_img=compressed_original_img_url)
        else:
            return "No file was uploaded.", 400
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return render_template('download.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)