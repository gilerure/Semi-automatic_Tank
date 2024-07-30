import cv2
import numpy as np
import torch
import os
import joblib
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import time

# 设置路径和模型
save_path = "faceinfo"
os.makedirs(save_path, exist_ok=True)
yolo_net = YOLO("yolov8n-face.pt")
pose_model = YOLO("yolov8n-pose.pt")
face_net = InceptionResnetV1(pretrained='vggface2').eval()

svm_model = joblib.load('best_svm_model_prob.joblib')
action_labels = ["attack", "defend", "normal"]

##############################################################################
###############################脸部工程########################################

# 图像预处理
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    preprocessed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return preprocessed

# 检测并返回第一个人脸
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

# 添加新的人脸到文件
def add_face_to_database(face_id, feature, category="neutral"):
    face_file = os.path.join(save_path, f"{face_id}.npy")
    if os.path.exists(face_file):
        data = np.load(face_file, allow_pickle=True).item()
        data['features'].append(feature)
    else:
        data = {"id": face_id, "category": category, "features": [feature]}
    np.save(face_file, data)

# 计算平均特征
def average_feature(features):
    return np.mean(features, axis=0)

# 匹配人脸
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
        
# 更新人物状态的函数
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
        np.save(face_file, data)

def track(box):
    x1, y1, width, height = box
    box_center_x = x1 + width / 2
    box_center_y = y1 + height / 2    
    
    n = 5
    
    if box_center_x < screen_width * (n / 2) / n:
        return 'left'
    elif box_center_x > (n / 2 + 1) * screen_width / n:
        return 'right'
    else:
        if box_center_y < screen_height * (n / 2) / n:
            return 'top'
        elif box_center_y > (n / 2 + 1) * screen_height / n:
            return 'down'
    return 'center'

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头，可以使用1、2等
last_time = 0  # 初始化上次检测的时间

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧")
        break
    
    screen_height = frame.shape[0]
    screen_width = frame.shape[1]
    
    # 图像预处理
    preprocessed_frame = preprocess_image(frame)

    # 检测并提取第一个人脸特征
    box = detect_first_face(preprocessed_frame)
    
    if box is not None:
        feature = extract_feature(preprocessed_frame, box)
        
        # 获取当前存储的人脸ID数量
        existing_ids = [int(f.split('.')[0]) for f in os.listdir(save_path) if f.endswith('.npy')]
        next_face_id = max(existing_ids) + 1 if existing_ids else 0

        # 匹配并添加新的人脸
        face_id, category = match_face(feature)
        if face_id is not None:
            print(f"识别出: {face_id} 类别: {category}")
            cv2.putText(frame, f"ID: {face_id} ({category})", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            add_face_to_database(face_id, feature)  # 更新特征
        else:
            add_face_to_database(next_face_id, feature)
            print(f"添加新的人脸: {next_face_id}")
            cv2.putText(frame, f"New ID: {next_face_id} (neutral)", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            category = "neutral"  # 默认类别为 neutral
            face_id = next_face_id

        # 根据类别选择框的颜色
        color = (255, 0, 0)  # 默认蓝色
        if category == "enemy":
            color = (0, 0, 255)  # 红色
        elif category == "friend":
            color = (0, 255, 0)  # 绿色

        # 显示检测结果
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)

        current_time = time.time()
        if current_time - last_time >= 1.5:  # 每1.5秒检测一次动作
            last_time = current_time  # 更新上次检测的时间

            # 提取关键点
            keypoints = extract_keypoints(frame)
            keypoints = keypoints.reshape(1, -1)  # 调整形状以匹配SVM模型输入

            # 预测动作
            prediction = svm_model.predict_proba(keypoints)  # 获取每个动作的概率
            action_index = np.argmax(prediction)  # 找到最大概率的索引
            action_prob = prediction[0][action_index]  # 获取最大概率值
            action = action_labels[action_index] if action_prob > 0.8 else "normal"
            
            if action != "normal":
                update_category(face_id, action)
        # 显示结果在帧上
        cv2.putText(frame, f'Predicted Action: {action}\n', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.imshow('Action Recognition', frame)

        direction = track(box=box)
        cv2.putText(frame, f'Please move your head to the {direction}.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 显示图像
    cv2.imshow('Face Detection and Recognition', frame)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
