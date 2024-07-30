from flask import Flask, render_template, Response, request
from picamera2 import Picamera2
import cv2
import time
import serial
import paho.mqtt.client as mqtt
import json
from PIL import Image
import base64
import io
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
socketio = SocketIO(app)

res_config = {"main": {"format": 'RGB888', "size": (1440, 1080)}}

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(**res_config))
picam2.start()

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.flush()  # 清空串口缓冲区
time.sleep(2)
ser.write(b'SYN\n')
print("Sent: SYN")

syn_ack = ser.readline().decode('utf-8').strip()
if syn_ack == "SYN-ACK":
    print("Received: SYN-ACK")
    ser.write(b'ACK\n')
    print("Sent: ACK")
    established = ser.readline().decode('utf-8').strip()
    if established == "Connection Established":
        print("Handshake successful, connection established")
else:
    print("Handshake failed")

mqtt_broker = "127.0.0.1"
mqtt_port = 1883
mqtt_topic_predict = "Group14/IMAGE/predict"
mqtt_topic_video = "Group14/IMAGE/video"

chase_mode = False
stop_movement = False
target_in_range = False

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker.")
        client.subscribe(mqtt_topic_predict)
    else:
        print("Failed to connect, return code %d\n", rc)

def on_message(client, userdata, msg):
    global chase_mode, stop_movement, target_in_range
    print(f"Received message from {msg.topic}: {msg.payload.decode()}")
    message = msg.payload.decode()
    if chase_mode:
        if "left" in message:
            ser.write(b'a\n')
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {response}")
        elif "right" in message:
            ser.write(b'd\n')
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {response}")
        elif "up" in message:
            ser.write(b'q\n')
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {response}")
        elif "down" in message:
            ser.write(b'e\n')
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {response}")
    if target_in_range and "centre" in message:
        ser.write(b'r\n')
        response = ser.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")
        socketio.start_background_task(target=mission_check)
    socketio.emit('mqtt_message', message)

client = mqtt.Client()
client.username_pw_set("andy", "040908")
client.on_connect = on_connect
client.on_message = on_message
client.connect(mqtt_broker)
client.loop_start()

def gen_frames():
    while True:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        img_str = base64.b64encode(frame).decode("utf-8")
        send_dict = {"data": img_str}
        client.publish(mqtt_topic_video, json.dumps(send_dict))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command', methods=['POST'])
def command():
    global chase_mode, stop_movement
    command = request.form.get('command')
    if chase_mode:
        if command == 'o':
            chase_mode = False
            stop_movement = False
            ser.write(b'x\n')
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {response}")
            socketio.emit('mode_status', {'chase_mode': chase_mode})
            print("Chase mode disabled")
        return '', 204

    if command in ['w', 'a', 's', 'd', 'x', 'q', 'e', 'r']:
        ser.write((command + '\n').encode('utf-8'))
        response = ser.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")
    elif command == 'o':
        chase_mode = True
        stop_movement = False
        print("Chase mode enabled")
        threading.Thread(target=distance_check).start()
        socketio.emit('mode_status', {'chase_mode': chase_mode})
    return '', 204

def distance_check():
    global chase_mode, stop_movement, target_in_range
    while chase_mode:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("Ping:"):
                distance = int(line.split()[1].replace("cm", ""))
                socketio.emit('distance_update', {'distance': distance})
                if distance > 10:
                    ser.write(b'w\n')
                    response = ser.readline().decode('utf-8').strip()
                    print(f"Arduino response: {response}")
                    stop_movement = False
                    target_in_range = False
                else:
                    ser.write(b'x\n')
                    response = ser.readline().decode('utf-8').strip()
                    print(f"Arduino response: {response}")
                    stop_movement = True
                    target_in_range = True
        time.sleep(0.1)

def mission_check():
    global chase_mode
    while chase_mode:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line == "Mission accomplished.":
                chase_mode = False
                stop_movement = False
                ser.write(b'x\n')
                response = ser.readline().decode('utf-8').strip()
                print(f"Arduino response: {response}")
                socketio.emit('mode_status', {'chase_mode': chase_mode})
                print("Chase mode disabled")
                break
        time.sleep(0.1)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
