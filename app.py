from flask import Flask , render_template ,request,Response
import requests
from inference import * 
import os
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom','best.pt', force_reload=True)

app = Flask(__name__)


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


###not done yet searching for method to return video as flask reponse

@app.route('/Detect_on_video',methods=['POST'])
def predict():
    if request.method == 'POST':
        video= request.files['video']
        file_path = os.path.join('uploads/' ,video.filename)
        video.save(file_path)
        result= Detect_On_Video(file_path,model)
        _, img_encoded = cv2.imencode('.png', result)
        response = img_encoded.tostring()
        os.remove(file_path)
    return Response(response=response, status=200, mimetype='image/png')


if __name__=='__main__':
    app.run()


