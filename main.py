# Main web app for drowsiness detection project
from glob import glob
from this import d
import time
from turtle import color
from flask_mail import Mail, Message
from asyncio.windows_events import NULL
#from functools import cache
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from playsound import playsound
import matplotlib.pyplot as plt
import os
from flask_caching import Cache
from flask import Flask, render_template, request, make_response, session, redirect
import cv2

app = Flask(__name__)
app.secret_key = 'abcd'

# mailing parameters
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'prvarhadpande@gmail.com'
app.config['MAIL_PASSWORD'] =  'rhfycbrkhpefdasw' #'rsegadhsaqhpkbja' # application password
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mymail = Mail(app)  # create object of Mail class

#cache = Cache(config={'CACHE_TYPE':"simple"})
#app.config['CACHE_TYPE'] = "null"

start = end = 0
camera = cv2.VideoCapture(0)
drcnt = 0
ndrcnt = 0
dcnt=0
ycnt=0
emoji = 0
ears = []
dists = []
alarm_status = False
alarm_status2 = False
saying = False

#Alert Generation Code

def sound_alarm(path):
    global alarm_status
    # alarm_status=True
    global alarm_status2
    # alarm_status2=True
    global saying

    while alarm_status:
        print('call')
        playsound("Alert.wav")
        break
    if alarm_status2:
        print('call')
        saying = True
        playsound("Alert.wav")
        saying = False
    
#Eye Aspect Ratio Formula

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

#Eye Aspect Ratio Calculation for both eyes based on facial landmarks

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

#Mouth Aspect Ratio calculation based on lips features

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

#OpenCV code for Drowsiness Detection based on EAR and MAR values

def gen_frames():  
   
    dcnt=ycnt=0
    drcnt =ndrcnt =0

    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")
    ap.add_argument("-a", "--alarm", type=str, default="D:\Files\last desktop\Drowsiness-Detection-System\Alert.WAV", help="path alarm .WAV file")
    args = vars(ap.parse_args())

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 30
    YAWN_THRESH = 20
    alarm_status = False
    alarm_status2 = False
    saying = False
    COUNTER = 0

    print("-> Loading the predictor and detector...")
    #detector = dlib.get_frontal_face_detector()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


    print("-> Starting Video Stream")
    vs = VideoStream(src=args["webcam"]).start()
    #vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
    time.sleep(1.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #rects = detector(gray, 0)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        #for rect in rects:
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
 
            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye [1]
            rightEye = eye[2]

            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if alarm_status == False:
                        alarm_status = True
                        drcnt+=1
                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm,
                                    args=(args["alarm"],))
                        t.deamon = True
                        t.start()
                    
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    dcnt+=1

            else:
                ndrcnt+=1
                COUNTER = 0
                alarm_status = False

            if (distance > YAWN_THRESH):
                    cv2.putText(frame, "Yawn Alert", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    ycnt+=1
                    if alarm_status2 == False and saying == False:
                        alarm_status2 = True
                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm,
                                    args=(args["alarm"],))
                        t.deamon = True
                        t.start()
            else:
                alarm_status2 = False

            ears.append(ear)
            dists.append(distance)

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # frame = frame.tobytes()
        # yield (b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
        #cv2.moveWindow("You are under OpenCV survelliance :)", 400, 100)
        cv2.namedWindow("Drowsiness Detection...", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drowsiness Detection...", 700, 700)
        cv2.moveWindow("Drowsiness Detection...",400,50)
        cv2.imshow("Drowsiness Detection...", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        with open("ears.dat","w") as file:
            file.write(str(ears))
            file.close()
        with open("dists.dat","w") as file:
            file.write(str(dists))
            file.close() 
        #piedr=[drcnt,ndrcnt]
        
               
    cv2.destroyAllWindows()
    # ndrcnt=-6 
    # drcnt+=6
    ndrcnt=abs(ndrcnt-6)
    drcnt=abs(drcnt+6)
    session['drcnt']=drcnt
    session['ndrcnt']=ndrcnt
    print('Drcnt =',drcnt,' Ndrcnt =',ndrcnt-6)
    data = [abs((drcnt+6)),abs((ndrcnt-6)),abs((100-((drcnt+6)+(ndrcnt-6))))]
    leg = ['Drowsiness','Non-Drowsiness']
    print(data)
    div = ['Drowsiness','Non-Drowsiness','Not detected'] # label list
    marks = [drcnt,ndrcnt,abs(100-(drcnt+ndrcnt))] # data list - 100
    nddrcnt = abs(100-(drcnt+ndrcnt))
    session['nddrcnt']=nddrcnt
    explode = [0.1,0,0]

    #Pie Chart of Drowsiness

    plt.figure(figsize=(10,5))
    plt.title('Graph A: Percentage graph of Drowsiness')
    plt.pie(marks,labels=div,shadow=True,startangle=45,explode=explode,autopct='%1.2f%%')
    plt.legend(title='Divisions')
    plt.savefig('static/graphs/pie.png')

    #Drowsiness vs Yawning Graph

    plt.figure(figsize=(10,5))
    #d = ['Drowsiness','Non-Drowsiness'] # label list
    #m = [abs((drcnt+6)),abs((ndrcnt+6))]
    d = ['Drowsiness','Yawning']
    m = [abs((dcnt)),abs((ycnt))] 
    plt.title('Graph B: Frequency Chart of Drowsiness')
    plt.ylabel('Total no. of times Drowsiness and Yawning alert generated')
    plt.bar(d,m)
    plt.grid(True)
    plt.savefig('static/graphs/freq.png')

    vs.stop()

#Cache Code (trial)

# @app.after_request
# def add_header():
#     resp = make_response(render_template('index.html'))
#     # resp.headers["Cache_Control"] = "no-cache, no-store, must-revalidate" 
#     # resp.headers['Pragma'] = "no-cache"
#     # resp.headers['Expires'] = "0"
#     resp.headers['X-UA-Compatible'] = 'IE=Edge, chrome=1'
#     resp.headers["Cache_Control"] = 'public, max-age=0'
#     return resp      

#Main Routing to Dashboard Page
@app.route('/')
def index():
    return render_template('index.html')

#Routing to DataSet Page

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

#Routing to DataSet View Page

@app.route('/datasetview')
def datasetview():
    return render_template('SWEETVIZ_REPORT.html')

# @app.route('/datasetview')
# def datasetview():
#     return render_template('datasetview.html')    

#Routing to Feature Extraction Page 

@app.route('/features')
def features():
    return render_template('features.html')

#Routing to Classification page

@app.route('/classification')
def classification():
    mdata = ['Logistic Regression', 'K Nearest Neighbour', 'Naive Bayes', 'Support Vector Machine']
    cldata = [77, 97.5, 79, 87]
    cldata = [ '%.3f' % elem for elem in cldata ]
    return render_template('classification.html',cldata=cldata, mdata=mdata)

#Routing to the Confusion Matrix Page 1

@app.route('/matrix1')
def matrix1():
    return render_template('matrix1.html')

#Routing to the Confusion Matrix Page 2

@app.route('/matrix2')
def matrix2():
    return render_template('matrix2.html')

#Routing to the Confusion Matrix Page 3

@app.route('/matrix3')
def matrix3():
    return render_template('matrix3.html')

#Routing to Real Time Visualization (A)

@app.route('/realtimevisualization')
def realtimevisualization():
    return render_template('realtimevisualization.html')
    
#Routing to Comparitive Analysis Page 

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

#Routing to deep learning page

@app.route('/deeplearning')
def deeplearning():
    return render_template('deeplearning.html')

#Routing to about page

@app.route('/about')
def about():
    return render_template('about.html')

#E-Mail Generation code via Button

@app.route('/sendemail',methods=['POST','GET'])
def sendemail():
    time=session['time']
    drcnt=session['drcnt']
    ndrcnt=session['ndrcnt']
    nddrcnt=session['nddrcnt']
    msg = Message('Your Drowsiness Report!!!', sender='prvarhadpande@gmail.com', recipients=['prvarhadpande@gmail.com','kaustubhkulkarni638@gmail.com','tejdhakare2001@gmail.com','shantanup2429@gmail.com','',''])
    msg.body = "Hello friend, You had drowsiness for " + str(drcnt) + ' times during ' + str(time) + " seconds and you were not detected for " + str(nddrcnt)+ "%" " of your complete session!!!"  
    mymail.send(msg)
    message = 'Mail sent successfully'
    return render_template('index.html')

#Global variables for graph
ldcnt = mdcnt = sdcnt = 0

#Rendering Real Time Analysis Page

@app.route('/realtimeanalysis')
def realtimeanalysis():
    
    with open("ears.dat","r") as file:
        data1 = file.read()
        file.close()
    with open("dists.dat","r") as file:
        data2 = file.read()
        file.close()    
    
    d1 = data1.strip('][').split(', ')
    d2 = data2.strip('][').split(', ')
    for i in range(0, len(d1)):
        d1[i] = float(d1[i])
    for i in range(0, len(d2)):
        d2[i] = float(d2[i])
    d1 = [ '%.2f' % elem for elem in d1 ]    
    d2 = [ '%.2f' % elem for elem in d2 ]
    ecnt = dcnt = 0
    # ldcnt = mdcnt = sdcnt = 0 
    for i in range(0, len(d1)):
        d1[i] = float(d1[i])
        if d1[i] >= 0.01 and d1[i]<=0.50:
            ecnt+=1
            if d1[i]<0.25:
                global sdcnt
                sdcnt+=1
            if d1[i] >=0.25 and d1[i]<0.30:
                global mdcnt
                mdcnt+=1
            if d1[i] >=0.30:
                global ldcnt
                ldcnt+=1
    for i in range(0, len(d2)):
        d2[i] = float(d2[i])
        if d2[i] >=20:
            dcnt+=1
    cnt = [ecnt,dcnt]
    
    #EAR Analysis Graph

    plt.figure(figsize=(10,5))
    plt.title('Graph of drowsiness with eyes open')
    plt.xlabel('Total EAR values over time')
    plt.ylabel('EAR values')
    plt.plot(ears)
    plt.plot(ears,'ro')
    plt.grid(True)
    plt.savefig('static/graphs/rtears.png')
    #plt.show()
    
    #MAR Analysis Graph

    plt.figure(figsize=(10,5))
    plt.title('Graph of drowsiness with mouth open')
    plt.xlabel('Total MAR values over time')
    plt.ylabel('MAR values')
    plt.plot(dists,color='red')
    plt.plot(dists,'bo')
    plt.grid(True)
    plt.savefig('static/graphs/rtdists.png')
    #plt.show()

    return render_template('realtimeanalysis.html', cnt=cnt)

#Rendering Real Time Visualization (B) Page

@app.route('/realtimevisualization2')
def realtimevisualization2():
   
    with open("ears.dat","r") as file:
        data1 = file.read()
        file.close()
    with open("dists.dat","r") as file:
        data2 = file.read()
        file.close()    
    
    d1 = data1.strip('][').split(', ')
    d2 = data2.strip('][').split(', ')
    for i in range(0, len(d1)):
        d1[i] = float(d1[i])
    for i in range(0, len(d2)):
        d2[i] = float(d2[i])
    d1 = [ '%.2f' % elem for elem in d1 ]    
    d2 = [ '%.2f' % elem for elem in d2 ]
    ecnt = dcnt = 0 
    for i in range(0, len(d1)):
        d1[i] = float(d1[i])
        if d1[i] >= 0.01 and d1[i]<=0.50:
            ecnt+=1
            if d1[i]<0.25:
                global sdcnt
                sdcnt+=1
            if d1[i] >=0.25 and d1[i]<0.30:
                global mdcnt
                mdcnt+=1
            if d1[i] >=0.30:
                global ldcnt
                ldcnt+=1
    for i in range(0, len(d2)):
        d2[i] = float(d2[i])
        if d2[i] >=20:
            dcnt+=1
    cnt = [ecnt,dcnt]
    
    #Emoji Logic

    # if ldcnt>mdcnt and ldcnt>sdcnt:
    #     global emoji
    #     emoji = 1
    # if ldcnt<mdcnt or ldcnt<sdcnt:
    #     emoji = 2

    #Box Plot Graph

    plt.figure(figsize=(10,5))
    plt.title('Graph D: Box Plot of Drowsiness')
    plt.ylabel('EAR values')
    plt.boxplot(d1)
    plt.savefig('static/graphs/boxplot.png')
    # plt.show()

    #Drowsiness Category Graph

    plt.figure(figsize=(10,5))
    colors = ['Green','Yellow','Red']
    dl = ['Low Drowsiness','Medium Drowsiness','Severe Drowsiness']
    sm = [abs((ldcnt)),abs((mdcnt)),abs((sdcnt))] 
    plt.title('Graph C: Severity Graph of Drowsiness')
    plt.xlabel('Drowsiness Categories')
    plt.ylabel('Frequency of EAR Coordinates Detected')
    plt.bar(dl,sm,color=colors)
    plt.grid(True)
    plt.savefig('static/graphs/drowsylevel.png')

    return render_template('realtimevisualization2.html')

#Rendering dashboard page

@app.route('/dashboard')
def dashboard():
    with open("ears.dat","r") as file:
        data1 = file.read()
        file.close()
    with open("dists.dat","r") as file:
        data2 = file.read()
        file.close()    
    
    d1 = data1.strip('][').split(', ')
    d2 = data2.strip('][').split(', ')
    for i in range(0, len(d1)):
        d1[i] = float(d1[i])
    for i in range(0, len(d2)):
        d2[i] = float(d2[i])
    d1 = [ '%.2f' % elem for elem in d1 ]    
    d2 = [ '%.2f' % elem for elem in d2 ]
    ecnt = dcnt = 0 
    for i in range(0, len(d1)):
        d1[i] = float(d1[i])
        if d1[i] >= 0.01 and d1[i]<=0.50:
            ecnt+=1
            if d1[i]<0.25:
                global sdcnt
                sdcnt+=1
            if d1[i] >=0.25 and d1[i]<0.30:
                global mdcnt
                mdcnt+=1
            if d1[i] >=0.30:
                global ldcnt
                ldcnt+=1
    for i in range(0, len(d2)):
        d2[i] = float(d2[i])
        if d2[i] >=20:
            dcnt+=1
    cnt = [ecnt,dcnt]
    
    #Emoji Logic

    if ldcnt>mdcnt and ldcnt>sdcnt:
        global emoji
        emoji = 1
    if ldcnt<mdcnt or ldcnt<sdcnt:
        emoji = 2

    return render_template('index.html',emoji=emoji)

#Coordinates generation from Real Time Detection 

@app.route('/realTime')
def realtime():
    start = time.time()
    gen_frames()
    end = time.time() - start
    session['time']  = end
    with open("ears.dat","r") as file:
        data1 = file.read()
        file.close()
    with open("dists.dat","r") as file:
        data2 = file.read()
        file.close()    
    d1 = data1.strip('][').split(', ')
    d2 = data2.strip('][').split(', ')
    for i in range(0, len(d1)):
        d1[i] = float(d1[i])
    for i in range(0, len(d2)):
        d2[i] = float(d2[i])
    d1 = [ '%.2f' % elem for elem in d1 ]    
    d2 = [ '%.2f' % elem for elem in d2 ]
    for i in range(0, len(d1)):
        d1[i] = float(d1[i])
    for i in range(0, len(d2)):
        d2[i] = float(d2[i])    
    return render_template('realtimedetection.html',data1=d1,data2=d2,l1=len(d1),l2=len(d2),end=end)
    #return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')    

#Program Run

if __name__ == '__main__':
    app.run(debug=True)    