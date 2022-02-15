import cv2
import dlib 
#read the image
img = cv2.imread("../face_recognition/a.jpeg")

#convert img to grayscale: 3D -> 2D
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

#dlib: load face recognition detector
face_detector = dlib.get_frontal_face_detector()

#load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#use detector to find face landmarks
faces = face_detector(gray)

for face in faces:
    x1 = face.left() #left point
    y1 = face.top()#top point
    x2 = face.right()# right point
    y2 = face.bottom()#bottom point

    #draw rectangle
    cv2.rectangle(img=img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=3)
    face_features = predictor(image=gray, box=face)

    #loop throungh all 68 points
    for i in range(68):
        x = face_features.part(i).x
        y = face_features.part(i).y

        #draw circle
        cv2.circle(img=img, center=(x,y), radius=2, color=(0,0,255), thickness=1)

#show the image
cv2.imshow(winname='Face Recognition App', mat=img)

#wait for a key press to exit
cv2.waitKey(delay=0)

#close all window
cv2.destroyAllWindows()