import torch 
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom','best.pt', force_reload=True)

def Detect_On_Image(img_path,model):
        test_img=cv2.imread(img_path)
        results=model(test_img)
        res= np.squeeze(results.render())
        cv2.imwrite("Results/res.jpg",cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
        return cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

def Detect_On_Video(video_path,model,webcam=False):
    if webcam:
        cap=cv2.VideoCapture(0)
    else:
        cap=cv2.VideoCapture(video_path)
    outputVideo= cv2.VideoWriter('Results/result.mov', 
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    10, (1020,600))
    while True:
            ret,frame=cap.read()
            if not ret:
                    print("no return")
                    break
            frame=cv2.resize(frame,(1020,600))
            results=model(frame)
            res=np.squeeze(results.render())
            outputVideo.write(res)
           
    cap.release()
    cv2.destroyAllWindows()

    
if __name__=='__main__':
    #Detect_On_Image("Tests/test1.jpg",model)
    Detect_On_Video("Tests/test_video.mov",model,webcam=True)






