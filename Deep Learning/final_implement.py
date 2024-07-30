from flask import Flask, Response, request, jsonify, render_template
import cv2
import numpy as np
import torch
import base64
import requests
import time
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import joblib
import os
from flask_socketio import SocketIO, emit
import threading  # 导入 threading 模块

app = Flask(__name__)
socketio = SocketIO(app)

# 设置类别名称
action_labels = ["attack", "defend", "normal"]

# 加载模型和路径
save_path = "faceinfo"
os.makedirs(save_path, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 预加载模型8 
yolo_net = YOLO("yolov8n_100e.pt")
pose_model = YOLO("yolov8n-pose.pt")
face_net = InceptionResnetV1(pretrained='vggface2').eval()
svm_model = joblib.load('best_svm_model_prob.joblib')

# 对方服务器的 URL
target_server_url = "http://172.25.96.203:5000"

# 设置处理间隔
process_interval = 0.3  # 处理间隔为0.5秒
action_check_interval = 2 # 每2秒检测一次动作

##############################################################################
###############################脸部工程########################################

# 处理为灰度图
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    preprocessed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return preprocessed

# 预测第一张脸
def detect_first_face(image, confidence_threshold=0.5):
    results = yolo_net(image)
    for result in results:
        for box in result.boxes:
            if box.conf[0] > confidence_threshold:  # 置信度阈值
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                return [x1, y1, width, height]
    return None

# 提取特征
def extract_feature(image, box):
    x, y, w, h = box
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    face = (face - 0.5) / 0.5  # 归一化
    with torch.no_grad():
        feature = face_net(face).cpu().numpy()
    return feature

# 添加到数据库
def add_face_to_database(face_id, feature, category="neutral"):
    face_file = os.path.join(save_path, f"{face_id}.npy")
    if os.path.exists(face_file):
        data = np.load(face_file, allow_pickle=True).item()
        data['features'].append(feature)
    else:
        data = {"id": face_id, "category": category, "features": [feature]}
    np.save(face_file, data)

# 展平特征
def average_feature(features):
    return np.mean(features, axis=0)

# 匹配脸部特征
def match_face(feature, threshold=0.6):
    max_similarity = 0
    matched_id = None
    matched_category = None
    for file in os.listdir(save_path):
        if file.endswith(".npy"):
            face_file = os.path.join(save_path, file)
            data = np.load(face_file, allow_pickle=True).item()
            avg_feature = average_feature(np.array(data['features']))
            similarity = cosine_similarity(feature, avg_feature.reshape(1, -1))
            if similarity > max_similarity:
                max_similarity = similarity
                matched_id = data["id"]
                matched_category = data["category"]
    if max_similarity > threshold:
        return matched_id, matched_category
    else:
        return None, None

##############################################################################
###############################姿势工程########################################

# 提取人的骨架
def extract_keypoints(image):
    results = pose_model(image)
    keypoints_list = []
    for result in results:
        if hasattr(result, 'keypoints'):
            keypoints = result.keypoints.xy.cpu().numpy()  # 获取关键点信息并转换为numpy数组
            keypoints_list.append(keypoints.flatten())  # 展平关键点
    if keypoints_list:
        # 确保所有关键点数组形状一致，这里假设每个人有17个关键点，每个关键点有x和y两个坐标
        expected_length = 34
        keypoints_array = [k for k in keypoints_list if len(k) == expected_length]
        if keypoints_array:
            return np.mean(keypoints_array, axis=0)
        else:
            return np.zeros(expected_length)
    else:
        return np.zeros(34)  # 假设每个人有17个关键点，每个关键点有x和y两个坐标

##############################################################################
###############################融合工程########################################

# 更新人的类别
def update_category(face_id, action):
    face_file = os.path.join(save_path, f"{face_id}.npy")
    if os.path.exists(face_file):
        data = np.load(face_file, allow_pickle=True).item()
        current_category = data["category"]
        if current_category == "neutral":
            if action == "attack":
                data["category"] = "enemy"
            elif action == "defend":
                data["category"] = "friend"
        elif current_category == "enemy" and action == "defend":
            data["category"] = "friend"
        elif current_category == "friend" and action == "attack":
            data["category"] = "enemy"
        np.save(face_file, data)

# 追踪方框的位置
def track(box, screen_width, screen_height):
    x1, y1, width, height = box
    box_center_x = x1 + width / 2
    box_center_y = y1 + height / 2    

    n = 3

    if box_center_x < screen_width * 1 / n:
        return 'left'
    elif box_center_x > (n -1) * screen_width / n:
        return 'right'
    else:
        if box_center_y < screen_height * 1 / n:
            return 'top'
        elif box_center_y > (n-1) * screen_height / n:
            return 'bottom'
    return 'centre'

def get_video_stream(url):
    stream = requests.get(url, stream=True)
    bytes_data = bytes()
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield img


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    threading.Thread(target=process_video_feed).start()
    return jsonify({"status": "processing started"})

def process_video_feed():
    video_url = "http://172.25.96.203:5000/video_feed"
    last_process_time = time.time()
    last_action_check_time = time.time() - action_check_interval  # 确保第一次进入循环时立即检测动作
    for frame in get_video_stream(video_url):
        current_time = time.time()
        if current_time - last_process_time >= process_interval:
            last_process_time = current_time

            preprocessed_frame = preprocess_image(frame)
            box = detect_first_face(preprocessed_frame)

            if box is not None:
                feature = extract_feature(preprocessed_frame, box)
                existing_ids = [int(f.split('.')[0]) for f in os.listdir(save_path) if f.endswith('.npy')]
                next_face_id = max(existing_ids) + 1 if existing_ids else 0

                face_id, category = match_face(feature)
                if face_id is not None:
                    add_face_to_database(face_id, feature)  # 更新特征
                else:
                    add_face_to_database(next_face_id, feature)
                    category = "neutral"  # 默认类别为 neutral
                    face_id = next_face_id

                if current_time - last_action_check_time >= action_check_interval:
                    last_action_check_time = current_time
                    keypoints = extract_keypoints(frame)
                    keypoints = keypoints.reshape(1, -1)  # 调整形状以匹配SVM模型输入

                    prediction = svm_model.predict_proba(keypoints)  # 获取每个动作的概率
                    action_index = np.argmax(prediction)  # 找到最大概率的索引
                    action_prob = prediction[0][action_index]  # 获取最大概率值
                    action = action_labels[action_index] if action_prob > 0.8 else "attack"

                    if action != "normal":
                        update_category(face_id, action)
                else:
                    action = "normal"

                screen_height = frame.shape[0]
                screen_width = frame.shape[1]
                direction = track(box, screen_width, screen_height)

                face_box_area = (box[2] * box[3]) / (screen_width * screen_height)  # 计算脸部框框在屏幕中的占比

                # 计算摄像头和脸部之间的距离

                response = {
                    "person_detected": True,
                    "face_id": face_id,
                    "action": action,
                    "category": category,
                    "direction": direction,
                    "face_box_area": face_box_area,
                }
            else:
                response = {
                    "person_detected": False,
                    "face_id": None,
                    "action": "normal",
                    "category": "none",
                    "direction": "none",
                    "face_box_area": 0,
                }

            print(f"Response: {response}")
            socketio.emit('response_message', response)

            # 将结果发送给对方服务器
            send_result_to_target_server(response)

def send_result_to_target_server(response):
    try:
        # 发送JSON数据到目标服务器（树莓派）
        target_server_url = "http://172.25.96.203:5000/process_video_feed"
        result = requests.post(target_server_url, json=response)
        if result.status_code == 200:
            print("Successfully sent result to target server via HTTP POST")
        else:
            print(f"Failed to send result to target server with status code {result.status_code}")
    except Exception as e:
        print(f"Error sending result to target server: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
