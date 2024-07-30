import time
import cv2
import serial
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
from picamera2 import Picamera2

app = Flask(__name__)
socketio = SocketIO(app)

res_config = {"main": {"format": 'RGB888', "size": (640, 640)}}

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(**res_config))
picam2.start()

server_url = "http://172.25.99.47:5000/process_video_feed"


def initialize_serial():
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        ser.flush()
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
                return ser
        else:
            print("Handshake failed")
            ser.close()
    except serial.SerialException as e:
        print(f"Error initializing serial connection: {e}")
        return None


ser = initialize_serial()

chase_mode = False
last_direction_command_time = time.time() - 1
target_in_range_count = 0
friend_count = 0


def gen_frames():
    while True:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)


def handle_server_response(response):
    global chase_mode, last_direction_command_time, target_in_range_count, friend_count
    stop_movement = True
    if response.get("person_detected"):
        category = response.get("category")
        direction = response.get("direction")
        face_box_area = response.get("face_box_area")
        if "friend" in category:
            friend_count += 1
            if friend_count >= 3:
                chase_mode = False
                target_in_range_count = 0
                friend_count = 0
                ser.write(b'x\n')
                ser.write(b'p\n')
                socketio.emit('mode_status', {'chase_mode': chase_mode})
                print("Chase mode disabled")
        if "enemy" in category and chase_mode:
            friend_count = 0
            if face_box_area < 0.01:
                ser.write(b'w\n')
                stop_movement = False
                target_in_range_count = 0
            else:
                stop_movement = True
                target_in_range_count += 1

            current_time = time.time()
            if current_time - last_direction_command_time >= 1:
                if "left" in direction:
                    ser.write(b'a\n')
                    last_direction_command_time = current_time
                elif "right" in direction:
                    ser.write(b'd\n')
                    last_direction_command_time = current_time

            if "top" in direction:
                ser.write(b'q\n')
            if "bottom" in direction:
                ser.write(b'e\n')

            if target_in_range_count >= 3 and "centre" in direction:
                ser.write(b'r\n')
                chase_mode = False
                target_in_range_count = 0
                ser.write(b'x\n')
                ser.write(b'p\n')
                socketio.emit('mode_status', {'chase_mode': chase_mode})
                print("Chase mode disabled")

        if stop_movement:
            ser.write(b'x\n')

    socketio.emit('server_response', response)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/command', methods=['POST'])
def command():
    global chase_mode
    command = request.form.get('command').lower()
    if chase_mode:
        if command == 'o':
            chase_mode = False
            ser.write(b'x\n')
            ser.write(b'p\n')
            socketio.emit('mode_status', {'chase_mode': chase_mode})
            print("Chase mode disabled")
        return '', 204

    if command in ['w', 'a', 's', 'd', 'x', 'q', 'e', 'r']:
        ser.write((command + '\n').encode('utf-8'))
    elif command == 'o':
        chase_mode = True
        ser.write(b'o\n')
        print("Chase mode enabled")
        socketio.emit('mode_status', {'chase_mode': chase_mode})
    return '', 204


@app.route('/process_video_feed', methods=['POST'])
def process_video_feed():
    response = request.get_json()
    print(f"Received POST request: {response}")
    if response:
        handle_server_response(response)
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"status": "failed"}), 400


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
