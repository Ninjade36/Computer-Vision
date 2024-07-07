from flask import Flask, request, render_template, send_from_directory, jsonify
from main import get_detections, process_image
import os
import torch
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

model_path = r'C:\\Users\\Adi\\Desktop\\fourthyear\\DL-Proj\\object-remove-main\\src\\models2\\best.pt'
model_custom = torch.hub.load(r'C:\Users\Adi\Desktop\fourthyear\DL-Proj\object-remove-main\yolov5', 'custom', path=model_path, source='local', force_reload=True)
model_custom.eval()  # Set the model to inference mode
yolov5_pretrained_weights_path = r'C:\Users\Adi\Desktop\fourthyear\DL-Proj\object-remove-main\yolov5\yolov5x.pt'
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_pretrained_weights_path, force_reload=True)
model_yolov5.eval()
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def resize_image(image_path, target_size=(512, 680), output_format='JPEG'):
    image = Image.open(image_path)
    image = ImageOps.contain(image, target_size, method=Image.Resampling.LANCZOS)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(image_path))
    image.save(output_path, format=output_format)
    return output_path

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    detection_type = request.form.get('detectionType', 'car')  # Default to car detection
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Choose the model based on detection_type
        if detection_type == 'general':
            model = model_yolov5  
        else:
            model = model_custom

        resized_image_path = resize_image(file_path)
        web_accessible_image_path = os.path.join('output', os.path.basename(resized_image_path))
        detections = get_detections(resized_image_path, model)
        return jsonify({'image_path': web_accessible_image_path, 'detections': detections})
    return jsonify({'error': 'Failed to upload file'}), 400


@app.route('/remove-object', methods=['POST'])
def remove_object():
    data = request.json
    image_path = data['image_path']
    selected_detection = data['selected_detection']
    print("hiii i am the image path :",image_path)
    processed_image_path = process_image(image_path, selected_detection)
    return jsonify({'processed_image_path': processed_image_path})

@app.route('/output/<filename>')
def serve_image(filename):
    """Serve images from the output directory."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
