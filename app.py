from flask import Flask , render_template ,request,Response
import requests
from inference import * 
import os
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom','best.pt', force_reload=True)

app = Flask(__name__)

@app.route('/')
def welcome_page():
    return render_template('index.html')


@app.route('/Detect_on_image',methods=['POST'])
def predict():
    if request.method == 'POST':
        image= request.files['image']
        file_path = os.path.join('uploads/' ,image.filename)
        image.save(file_path)
        result= Detect_On_Image(file_path,model)
        _, img_encoded = cv2.imencode('.png', result)
        response = img_encoded.tostring()
        os.remove(file_path)
    return Response(response=response, status=200, mimetype='image/png')


def generate_frames(file_path=' ',webcam=False):
    if webcam:
        cap=cv2.VideoCapture(0)
    else:
        cap=cv2.VideoCapture(file_path)
    while True:
            ret,frame=cap.read()
            if not ret:
                    print("no return")
                    break
            frame=cv2.resize(frame,(1020,600))
            results=model(frame)
            res=np.squeeze(results.render())
            _, img_encoded = cv2.imencode('.jpg', res)
            frame = img_encoded.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()


@app.route('/Detect_on_video',methods=['POST'])
def predict_():
    if request.method == 'POST':
        video= request.files['video']
        file_path = os.path.join('uploads/' ,video.filename)
        video.save(file_path)
    return  Response(generate_frames(file_path),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/Detect_on_Webcam',methods=['GET'])
def predict__():
    return  Response(generate_frames(webcam=True),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(debug=True)


