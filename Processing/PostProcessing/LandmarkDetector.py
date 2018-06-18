from imutils import face_utils
import dlib
import cv2
import os
from time import sleep

dataset_path = '../../../../datasets/cyber_extruder_ultimate'

if __name__ == '__main__':

    p = "shape_predictor_5_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    for root, dirs, files in os.walk(dataset_path):
        for direc in dirs:
            path = dataset_path + os.sep + direc + os.sep + '00000%d.jpg'
            sequence = cv2.VideoCapture(path)
            while True:
                ret, image = sequence.read()
                if image is None:
                    break
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # gray = image

                rects = detector(gray, 0)

                for (i, rect) in enumerate(rects):

                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    for (x, y) in shape:
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                cv2.imshow("Output", image)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    sequence.release()
                    cv2.destroyAllWindows()
                    quit()
