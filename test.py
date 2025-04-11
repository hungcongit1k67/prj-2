from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
from main import solve  # Import hàm solve từ file nguong_cung.py
from flask_cors import CORS  # Cho phép gọi API từ Android app

app = Flask(__name__)
CORS(app)  # Cho phép CORS

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    processed_image_path = os.path.join(PROCESSED_FOLDER, "processed_" + image_file.filename)

    image_file.save(image_path)
    image = cv2.imread(image_path)

    if image is None:
        return jsonify({"error": "Could not load image"}), 400

    try:
        processed_image, result = solve(image)
        cv2.imwrite(processed_image_path, processed_image)

        # ✅ Sửa URL để có đầy đủ scheme "http://"
        full_url = f"http://{request.host}/download/processed_{image_file.filename}"
        print("Processed Image URL:", full_url)
        return jsonify({
            "result": result,
            "processed_image_url": full_url  # ✅ Đảm bảo URL đúng định dạng
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='192.168.1.149', port=5000, debug=True)
