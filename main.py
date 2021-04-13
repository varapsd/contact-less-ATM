import tkinter as tk
import cv2
from PIL import Image,ImageTk
import face_recognition
import numpy as np
import dlib
from scipy.spatial import distance as dist
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time
from playsound import playsound
import pymongo
from datetime import date

LARGE_FONT = ("Verdana", 25)
cap = cv2.VideoCapture(0)
cap1 = cap

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3
EAR_AVG = 0



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

url = '''mongodb url'''
client = pymongo.MongoClient(url)
db = client.contactlessAtm

class application(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        F = PageOne
        frame = F(container,self)
        self.frames[F] = frame
        frame.grid(row=0,column=0, sticky="nsew")
        '''
        for F in (PageTwo,PageOne):
            print(F)
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")
        '''
        self.show_frame(PageOne)
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.userId = None
        self.userName = None
        self.COUNTER = 0
        self.TOTAL = 0
        label = tk.Label(self, text="Welcome to the Contact less ATM", font=LARGE_FONT, height=2, bg='skyblue')
        label.pack(pady=10, padx=10, fill=tk.X)
        self.controller = controller
        self.page = 0
        self.frame1 = tk.Frame(self)
        self.frame1.pack(side=tk.LEFT, padx=0, anchor=tk.NW)
        self.webcam = tk.Canvas(self.frame1, width=cv2.CAP_PROP_FRAME_WIDTH * 200,
                                height=cv2.CAP_PROP_FRAME_HEIGHT * 120)
        self.webcam.pack(padx=10)
        self.frame2 = tk.Frame(self)
        self.frame2.pack(side=tk.LEFT,padx=30,anchor=tk.N)
        self.data1 = tk.StringVar()
        self.data = ''
        self.data1.set(self.data)
        self.label1 = tk.Label(self.frame2,textvariable=self.data1,font=('Arial',20,'bold'),bd=15,bg='GreenYellow',width=65,pady=2,justify='center',height=12)
        self.label1.pack(padx=20)
        '''
        button1 = tk.Button(self, text="Back to Home",
                            command=lambda:self.changePage())
        button1.pack()
        '''

        self.passTimeInit = False
        self.passTimePrev = None
        self.passTimeValue = 0
        self.pin = ''
        self.startPage()

    def eye_aspect_ratio(self,eye):
        # compute the euclidean distance between the vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])

        # compute the EAR
        ear = (A + B) / (2 * C)
        return ear
    def startPage(self):
        self.data = "------------------------- Capturing Image ------------------------"
        self.data += "\n * Please keep face to the middle of the camera "
        self.data += "\n * Maintain necessary distance to capture the face properly "
        self.data += "\n * Please perform two eye blink operation "
        self.data1.set(self.data)
        self.update_image()

    def update_image(self):
        # Get the latest frame and convert image format

        self.load = cap.read()[1]
        self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB

        try:

            self.faceloc = face_recognition.face_locations(self.image)[0]
            cv2.rectangle(self.load, (self.faceloc[3], self.faceloc[0]),
                          (self.faceloc[1], self.faceloc[2]), (255, 0, 255), 2)
            # convert the frame to grayscale
            gray = self.image
            rects = detector(gray, 0)
            for rect in rects:
                x = rect.left()
                y = rect.top()
                x1 = rect.right()
                y1 = rect.bottom()
                # get the facial landmarks
                landmarks = np.matrix([[p.x, p.y] for p in predictor(self.load, rect).parts()])
                # get the left eye landmarks
                left_eye = landmarks[LEFT_EYE_POINTS]
                # get the right eye landmarks
                right_eye = landmarks[RIGHT_EYE_POINTS]
                # draw contours on the eyes
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(self.load, [left_eye_hull], -1, (0, 255, 0),
                                 1)  # (image, [contour], all_contours, color, thickness)
                cv2.drawContours(self.load, [right_eye_hull], -1, (0, 255, 0), 1)
                # compute the EAR for the left eye
                ear_left = self.eye_aspect_ratio(left_eye)
                # compute the EAR for the right eye
                ear_right = self.eye_aspect_ratio(right_eye)
                # compute the average EAR
                ear_avg = (ear_left + ear_right) / 2.0
                # detect the eye blink
                if ear_avg < EYE_AR_THRESH:
                    self.COUNTER += 1
                else:
                    if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1
                        print("Eye blinked")
                    COUNTER = 0

                cv2.putText(self.load, "Blinks{}".format(self.TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
                cv2.putText(self.load, "EAR {}".format(ear_avg), (10, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)

        except Exception as e:
            print('--waiting---')

        if self.TOTAL<2:
        # Update image
            self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(self.image)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)

            # Repeat every 'interval' ms
            self.webcam.after(10, self.update_image)
        else:
            self.load = cap.read()[1]
            self.image1 = self.load
            self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)
            if(len(face_recognition.face_locations(self.image))==0):
                TOTAL = 0
                self.webcam.after(10, self.update_image)
            else:
                cv2.putText(self.image,"Processing for indentification",(10,30),
                            cv2.FONT_HERSHEY_DUPLEX,0.7,(0,255,255),1)
                self.image = Image.fromarray(self.image)
                self.image = ImageTk.PhotoImage(self.image)
                self.webcam.create_image(0,0, anchor = tk.NW, image=self.image)

                self.persons = os.listdir("./images")
                for user in self.persons:
                    self.userImg = face_recognition.load_image_file('./images/' + user)
                    self.userImg = cv2.cvtColor(self.userImg, cv2.COLOR_BGR2RGB)
                    self.userEnc = face_recognition.face_encodings(self.userImg)[0]
                    self.ssEnc = face_recognition.face_encodings(self.load)[0]
                    self.results = face_recognition.compare_faces([self.userEnc], self.ssEnc)[0]
                    if (self.results):
                        self.userId = int(user.split('.')[0])
                        self.userName = db.users.find_one({'userId':self.userId},{"userName":1})['userName']
                        self.userPage0()

    def userPage0(self):
        playsound('./sounds/transtio.mp3')
        self.data = "------------------------- Capturing Image ------------------------"
        self.data += "\n * Welcome to the ATM " + self.userName
        self.data +="\n  * Please display your fingers at optimal posture to the"
        self.data +="\n   corresponding Security PIN of the account selected"
        self.data += "\n\nYour PIN : "
        self.data1.set(self.data)
        self.page0Cam()
    def userPage1(self):
        playsound('./sounds/transtio.mp3')
        self.accountsData = db.users.find_one({'userId':self.userId},{'accounts':1})['accounts']
        self.accounts = list(map(lambda x:x['accountId'],self.accountsData))
        self.data = "------------------------- Select Your Bank Account ------------------------\n"
        self.data +="\n*Please display your fingers at optimal posture to the"
        self.data +="\n   corresponding account to select\n\n"
        for i in range(len(self.accounts)):
            self.data += str(i+1)+' -' +str(self.accounts[i])+'\n\n'
        self.data += str(len(self.accounts)+1)+" Cancel Transaction"
        self.data1.set(self.data)
        self.page1Cam()
    def userPage2(self):
        playsound('./sounds/transtio.mp3')
        self.data = "------------------------- Select Menu ------------------------"
        self.data +="\n* Please display your fingers at optimal posture to the"
        self.data +="\n   corresponding option to select"
        self.data += "\n\n\n1. Balance Enquiry\n\n2.MINI Statement\n\n3.Balance withdraw\n\n4.Cancel Transaction"
        self.data1.set(self.data)
        self.page2Cam()

    '''
    def changePage(self):
        self.page += 1
        if(self.page == 1):
            self.data = "------------------------- Select Your Bank Account ------------------------"
            self.data+= "\n\n\n1. 123456789\n\n2.789456123\n\n3.456123789"
            self.data1.set(self.data)
            self.updatePage2()
        elif(self.page == 2):
            self.data = "------------------------- Select Menu ------------------------"
            self.data += "\n\n\n1. Balance Enquiry\n\n2.MINi Statement\n\n3.Balance withdraw"
            self.data1.set(self.data)
            self.button3 = tk.Button(self.frame2,text="balanceEnqury",command=lambda:self.balanceEnqury())
            self.button3.pack()
            self.button4 = tk.Button(self.frame2,text="miniStatement",command=lambda:self.miniStatement())
            self.button4.pack()
            self.button5 = tk.Button(self.frame2,text="withdraw",command=lambda:self.withdraw())
            self.button5.pack()
        else:
            self.page = 0
            self.changePage()
            '''
    def balanceEnqury(self):
        playsound('./sounds/transtio.mp3')
        balance = self.selectedAccount['accountBalance']
        self.data = "user balance is : " + str(balance)
        self.data1.set(self.data)
        self.label1.after(5000,lambda :self.data1.set('THANK YOU'))
        self.userId = None
        self.userName = None
        self.COUNTER = 0
        self.TOTAL = 0
        self.pin = ''
        self.label1.after(8000,self.startPage)

    def miniStatement(self):
        playsound('./sounds/transtio.mp3')
        self.data = "user MINI Statement"
        for i,tr in enumerate(self.selectedAccount['miniStatement']):
            print(tr['trDate'])
            self.data += '\n\n'+str(i)+" : "+str(tr['trAmount']) + ' ' + tr['trType'] + ' ' + tr['trDate']
        '''
        self.data += "\n1000 w 10/10/2020"
        self.data += "\n2000 c 1/11/2020"
        self.data += "\n10000 w 1/12/2020"
        '''
        self.data1.set(self.data)
        self.label1.after(10000,lambda : self.data1.set('THANK YOU'))
        self.userId = None
        self.userName = None
        self.COUNTER = 0
        self.TOTAL = 0
        self.pin = ''
        self.label1.after(14000,self.startPage)


    def withdraw(self):
        playsound('./sounds/transtio.mp3')
        #self.data = "---  withdraw Money---\n\nSelect Options"
        #self.data1.set(self.data)
        #self.data1.set('')
        self.label1.pack_forget()

        self.optionsFrame = tk.Frame(self.frame2)
        self.optionsFrame.pack(side=tk.BOTTOM, padx=0, anchor=tk.NW)
        tk.Label(self.optionsFrame,text = '--- Select Options ---',font=('Arial',25,'bold'),bd=15,
                 bg='Yellow',width=35,pady=2,justify='center',height=2).pack(padx=20)
        tk.Label(self.optionsFrame, text="1. 200\t2. 500", font=('Arial', 25), bd=15, bg='greenyellow', width=30,
                 pady=2, justify='center', height=1).pack(padx=50)
        tk.Label(self.optionsFrame, text="3. 1000\t4. 2000", font=('Arial', 25), bd=15, bg='skyblue', width=30, pady=2,
                 justify='center', height=1).pack(padx=50)
        tk.Label(self.optionsFrame, text="5. 5000\t6. 10000", font=('Arial', 25), bd=15, bg='greenyellow', width=30,
                 pady=2, justify='center', height=1).pack(padx=50)
        tk.Label(self.optionsFrame, text="7.Other Value", font=('Arial', 25), bd=15, bg='skyblue', width=30, pady=2,
                 justify='center', height=1).pack(padx=50)
        self.withdrawPage1Cam()

    def withdrawPage1Cam(self):
        self.load = cap.read()[1]
        totalFingers, img = self.countFingers(self.load)
        # print(totalFingers)
        self.flag = False
        if (totalFingers == self.passTimePrev and self.passTimePrev != None):
            if not self.passTimeInit:
                self.passTimeInit = True
                self.passTimeValue = time.time()
            if time.time() - self.passTimeValue >= 2:
                print(totalFingers, 'confirm')
                self.flag = True
                self.passTimeValue = time.time() + 10
                self.passTimePrev = None
                playsound('./sounds/capture.mp3')
                self.withdrawPage1Res(totalFingers)
                self.optionsFrame.destroy()
                self.label1.pack(padx=20)
                return

        else:
            self.passTimeInit = False
            self.passTimePrev = totalFingers
        if not self.flag:
            # self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(img)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.webcam.after(10, self.withdrawPage1Cam)

    def withdrawPage1Res(self,fingerCount):
        if(fingerCount == 1):
            self.withdrawConfirm(200)
        elif(fingerCount == 2):
            self.withdrawConfirm(500)
        elif(fingerCount == 3):
            self.withdrawConfirm(1000)
        elif(fingerCount == 4):
            self.withdrawConfirm(2000)
        elif(fingerCount == 5):
            self.withdrawConfirm(5000)
        elif(fingerCount == 6):
            self.withdrawConfirm(10000)
        elif(fingerCount == 7):
            self.withdrawPage2()
        else:
            self.data1.set('invalid Input\nTry Again')
            self.optionsFrame.destroy()
            self.label1.after(3000, self.withdraw)

    def withdrawPage2(self):
        playsound('./sounds/transtio.mp3')
        self.data = 'Enter the amount to withdraw\n'
        self.data +='\nPlease display 10 fingers to withdraw'
        self.data += '\n\nAmount : '
        self.data1.set(self.data)
        self.wAmount = ''
        self.withdrawPage2Cam()

    def withdrawPage2Cam(self):

        self.load = cap.read()[1]
        totalFingers, img = self.countFingers(self.load)
        # print(totalFingers)
        self.flag = False
        if (totalFingers == self.passTimePrev and self.passTimePrev != None):
            if not self.passTimeInit:
                self.passTimeInit = True
                self.passTimeValue = time.time()
            if time.time() - self.passTimeValue >= 2:
                print(totalFingers, 'confirm')
                self.flag = True
                self.passTimeValue = time.time() + 10
                playsound('./sounds/capture.mp3')
                self.passTimePrev = None
                self.withdrawPage2Res(totalFingers)
                return

        else:
            self.passTimeInit = False
            self.passTimePrev = totalFingers
        if not self.flag:
            # self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(img)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.webcam.after(10, self.withdrawPage2Cam)

    def withdrawPage2Res(self,fingerCount):
        if(fingerCount == 10):
            self.withdrawConfirm(int(self.wAmount))
        else:
            self.wAmount += str(fingerCount)
            self.data += str(fingerCount)
            self.data1.set(self.data)
            playsound('./sounds/button.mp3')
            self.withdrawPage2Cam()

    def withdrawConfirm(self,amount):
        playsound('./sounds/transtio.mp3')
        self.withAmount = amount
        self.data = "please confirm the withdraw Amount"
        self.data +="\n*Please display your fingers at optimal posture to the "
        self.data +="\ncorresponding option to select"
        self.data += "\n\nSelected Amount : "+str(amount)
        self.data += "\n\n1.Confirm\n2. Re-Select amount\n3. Cancel Transaction"
        self.data1.set(self.data)
        self.withdrawConfirmCam()

    def withdrawConfirmCam(self):
        self.load = cap.read()[1]
        totalFingers, img = self.countFingers(self.load)
        # print(totalFingers)
        self.flag = False
        if (totalFingers == self.passTimePrev and self.passTimePrev != None):
            if not self.passTimeInit:
                self.passTimeInit = True
                self.passTimeValue = time.time()
            if time.time() - self.passTimeValue >= 2:
                print(totalFingers, 'confirm')
                self.flag = True
                self.passTimeValue = time.time() + 10
                self.passTimePrev = None
                playsound('./sounds/capture.mp3')
                self.withdrawConfirmRes(totalFingers)
                return

        else:
            self.passTimeInit = False
            self.passTimePrev = totalFingers
        if not self.flag:
            # self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(img)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.webcam.after(10, self.withdrawConfirmCam)

    def withdrawConfirmRes(self,fingerCount):
        if(fingerCount == 1):
            self.debit(int(self.withAmount))
        elif(fingerCount == 2):
            self.withdraw()
        elif(fingerCount == 3):
            self.cancel()
        else:
            self.data1.set('invalid Input\nTry Again')
            self.label1.after(3000, self.cancel)



    def debit(self,debAmount):
        playsound('./sounds/transtio.mp3')
        curAmount = self.selectedAccount['accountBalance']
        if(curAmount < debAmount):
            result = '\n\n\nInsufficent balance Try again'
            self.data1.set(result)
            self.label1.after(2000, lambda: self.data1.set("\n\n\nTHANK YOU"))
            self.userId = None
            self.userName = None
            self.COUNTER = 0
            self.TOTAL = 0
            self.pin = ''
            self.label1.after(4000, self.startPage)
        else:
            qry1 = {"userId":self.userId,"accounts":{"$elemMatch": { "accountId" : self.selectedAccount['accountId']}}}
            updt1 = {"$set":{ "accounts.$.accountBalance": curAmount - debAmount }}
            x = db.users.update_one(qry1, updt1)
            today = date.today()
            print(today)
            curDate = today.strftime("%d/%m/%y")
            print(curDate)
            newTid = len(self.selectedAccount['miniStatement'])+1
            var = {
                "trId": len(self.selectedAccount['miniStatement'])+1,
                "trType": 'w',
                "trAmount": debAmount,
                "trDate": curDate
            }
            updt2 = { "$push":{"accounts.$.miniStatement" : var}}
            x = db.users.update_one(qry1, updt2)
            result = '\n\n\nTransaction Sucess\nPlease take you MONEY'
            self.data1.set(result)
            self.label1.after(2000,lambda : playsound('./sounds/withdraw1.mp3'))
            self.label1.after(12000,lambda : self.data1.set("\n\n\nTHANK YOU"))
            self.COUNTER = 0
            self.TOTAL = 0
            self.userId = None
            self.userName = None
            self.pin = ''
            self.label1.after(14000,self.startPage)

    def page0Cam(self):
        # Get the latest frame and convert image format
        self.load = cap.read()[1]
        totalFingers,img = self.countFingers(self.load)
        #print(totalFingers)
        self.flag = False
        if(totalFingers == self.passTimePrev and self.passTimePrev != None):
            if not self.passTimeInit:
                self.passTimeInit = True
                self.passTimeValue = time.time()
            if time.time() - self.passTimeValue >= 2:
                print(totalFingers,'confirm')
                self.flag = True
                self.passTimeValue = time.time()+10
                playsound('./sounds/capture.mp3')
                self.passUpdate(totalFingers)
                return

        else:
            self.passTimeInit = False
            self.passTimePrev = totalFingers
        if not self.flag:
            #self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(img)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)


        # Repeat every 'interval' ms
        if not self.flag:
            self.webcam.after(10, self.page0Cam)


        '''
        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = tk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(StartPage))
        button2.pack()
        '''

    def page1Cam(self):
        self.load = cap.read()[1]
        totalFingers, img = self.countFingers(self.load)
        # print(totalFingers)
        self.flag = False
        if (totalFingers == self.passTimePrev and self.passTimePrev != None):
            if not self.passTimeInit:
                self.passTimeInit = True
                self.passTimeValue = time.time()
            if time.time() - self.passTimeValue >= 2:
                print(totalFingers, 'confirm')
                self.flag = True
                self.passTimeValue = time.time() + 10
                self.passTimePrev = None
                playsound('./sounds/capture.mp3')
                self.accountSelect(totalFingers)
                return

        else:
            self.passTimeInit = False
            self.passTimePrev = totalFingers
        if not self.flag:
            # self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(img)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.webcam.after(10, self.page1Cam)

    def page2Cam(self):
        self.load = cap.read()[1]
        totalFingers, img = self.countFingers(self.load)
        # print(totalFingers)
        self.flag = False
        if (totalFingers == self.passTimePrev and self.passTimePrev != None):
            if not self.passTimeInit:
                self.passTimeInit = True
                self.passTimeValue = time.time()
            if time.time() - self.passTimeValue >= 2:
                print(totalFingers, 'confirm')
                self.flag = True
                self.passTimeValue = time.time() + 10
                self.passTimePrev = None
                playsound('./sounds/capture.mp3')
                self.actionSelect(totalFingers)
                return

        else:
            self.passTimeInit = False
            self.passTimePrev = totalFingers
        if not self.flag:
            # self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(img)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.webcam.after(10, self.page2Cam)

    def countFingers(self,image):
        with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            totalFinger = None
            if results.multi_hand_landmarks:
                totalFinger = 0
                fingers = [[2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
                i = 0
                for handss in results.multi_hand_landmarks:
                    if (handss.landmark[5].x < handss.landmark[9].x):
                        if (handss.landmark[fingers[0][0]].x > handss.landmark[fingers[0][1]].x and
                                handss.landmark[fingers[0][1]].x > handss.landmark[fingers[0][2]].x):
                            totalFinger += 1
                    else:
                        if (handss.landmark[fingers[0][0]].x < handss.landmark[fingers[0][1]].x and
                                handss.landmark[fingers[0][1]].x < handss.landmark[fingers[0][2]].x):
                            totalFinger += 1
                    for finger in range(1, 5):
                        if (handss.landmark[fingers[finger][0]].y > handss.landmark[fingers[finger][1]].y and
                                handss.landmark[fingers[finger][1]].y > handss.landmark[fingers[finger][2]].y and
                                handss.landmark[fingers[finger][2]].y > handss.landmark[fingers[finger][3]].y):
                            totalFinger += 1
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, str(totalFinger), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

                # cv2.putText(image, str(totalFinger), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
            return totalFinger,image
    def passUpdate(self,fingerCount):
        self.pin += str(fingerCount)
        if(fingerCount>9):
            self.page0Cam()
        print('--wait--')
        self.data += "X "
        self.data1.set(self.data)
        playsound('./sounds/button.mp3')
        if len(self.pin) == 4:
            self.authenticate(int(self.pin))
            #self.userPage1()
        else:
            self.page0Cam()

    def authenticate(self,entPin):
        userPin = db.users.find_one({'userId':self.userId},{'pin':1})['pin']
        if(userPin == entPin):
            self.userPage1()
        else:
            self.data = '\n\n\nInvalid Password TRY AGAIN'
            self.data1.set(self.data)
            self.pin = ''
            self.userId = None
            self.userName = None
            self.COUNTER = 0
            self.TOTAL = 0
            self.label1.after(3000,self.startPage)

    def accountSelect(self,fingerCount):
        if(fingerCount == len(self.accountsData)+1):
            self.cancel()
        if(fingerCount>0 and fingerCount<=len(self.accountsData)):
            self.selectedAccount = self.accountsData[fingerCount-1]
            print(self.selectedAccount)
            self.userPage2()
        else:
            self.data1.set('invalid Input\nTry Again')
            self.label1.after(3000, self.userPage1)

    def actionSelect(self,fingerCount):
        if(fingerCount == 1):
            self.balanceEnqury()
        elif(fingerCount == 2):
            self.miniStatement()
        elif(fingerCount == 3):
            self.withdraw()
        elif(fingerCount == 4):
            self.cancel()
        else:
            self.data1.set('invalid Input\nTry Again')
            self.label1.after(3000,self.userPage2)

    def cancel(self):
        playsound('./sounds/transtio.mp3')
        self.data = "\n\n\nTRANSCATION CANCELLED\n\nTHANK YOU"
        self.data1.set(self.data)
        self.pin = ''
        self.userId = None
        self.userName = None
        self.COUNTER = 0
        self.TOTAL = 0
        self.label1.after(3000, self.startPage)




app = application()
width = app.winfo_screenwidth()
height = app.winfo_screenheight()
app.geometry('%dx%d' % (width,height))
app.mainloop()
