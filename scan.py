import cv2
import threading
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

face_match_miles = False
face_match_kt = False
face_match_chris = False
face_match_josh = False
face_match_dev = False
face_match_moe = False


reference_img = cv2.imread("miles2.JPG")
reference_img2 = cv2.imread("moe.JPG")
reference_img3 = cv2.imread("chris.JPG")
reference_img4 = cv2.imread("josh.PNG")
reference_img5 = cv2.imread("dev.png")
reference_img6 = cv2.imread("kt.jpg")



def check_face(frame):
    global face_match_miles
    global face_match_chris
    global face_match_moe
    global face_match_josh
    global face_match_dev
    global face_match_kt
    global face_match

    try:
        if DeepFace.verify([frame, reference_img.copy()])['verified']:
            face_match_miles = True
        elif DeepFace.verify([frame, reference_img2.copy()])['verified']:
            face_match_moe = True
        elif DeepFace.verify([frame, reference_img3.copy()])['verified']:
            face_match_chris = True
        elif DeepFace.verify([frame, reference_img4.copy()])['verified']:
            face_match_josh = True
        elif DeepFace.verify([frame, reference_img5.copy()])['verified']:
            face_match_dev = True
        elif DeepFace.verify([frame, reference_img6.copy()])['verified']:
            face_match_kt = True
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face,args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match_miles:
            cv2.putText(frame, "Miles", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        if face_match_moe:
            cv2.putText(frame, "Moe", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        if face_match_dev:
            cv2.putText(frame, "Devin", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        if face_match_josh:
            cv2.putText(frame, "Josh", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        if face_match_chris:
            cv2.putText(frame, "Chris", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        if face_match_kt:
            cv2.putText(frame, "Katerian", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
