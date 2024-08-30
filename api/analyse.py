from flask import Flask, request, jsonify
import cv2
import numpy as np
from test import process_image

app = Flask(__name__)

@app.route('/process_fish', methods=['POST'])
def process_fish():
    # 读取上传的图像
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # 调用处理函数
    result = process_image(image)
    
    if isinstance(result, dict) and "error" in result:
        return jsonify(result), 400
    else:
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
