from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import traceback
import numpy as np
from collections import deque

from model.deep.FaceRecognizer import FaceRecognizer
from process.FaceDetector import FaceDetector
from process.post_process.FaceAlign import FaceAlign
from process.post_process.LandmarkDetector import LandmarkDetector
from utils.common import RepeatedTimer, clamp_rectangle


source = None  # reference to cv2.VideoCapture object
fps_counter = None  # repeated timer object
frame_count = 0  # frames ingested
fps = 0  # computed fps
sequence = 0  # sequence indicating consecutive face detections
landmarks = []  # list to hold the face landmarks across the batch
faces = []  # list to hold face bounding boxes across the batch
# queue holding information of the last fps counts; used to generate avg, fps
fps_queue = deque(maxlen=100)


def run():
    """
    Main loop of the program
    """

    global frame_count
    global source
    global fps_counter
    global fps
    global sequence
    global landmarks
    global faces

    face_area_threshold = 0.15
    camera_index = 0
    # width, height = 150, 150
    batch_size = 32
    face_recognition_confidence_threshold = 0.25

    face_landmark_predictor_path = "/media/ankurrc/new_volume/softura/facerec/code/face-trigger/pre_trained/shape_predictor_5_face_landmarks.dat"
    face_recognizer_model_path = "/media/ankurrc/new_volume/softura/facerec/code/face-trigger/pre_trained/dlib_face_recognition_resnet_model_v1.dat"
    svm_model_path = "/media/ankurrc/new_volume/softura/facerec/code/face-trigger/pre_trained/svm_proba.pkl"

    source = cv2.VideoCapture(index=camera_index)
    source.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
    source.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    print(source.get(cv2.CAP_PROP_FRAME_WIDTH), "x",
          source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_counter = RepeatedTimer(interval=2.0, function=fps_count)

    # reference to face detector
    face_detector = FaceDetector(face_area_threshold=face_area_threshold)
    # reference to landmark detector
    landmark_detector = LandmarkDetector(
        predictor_path=face_landmark_predictor_path)
    # reference to face recognizer
    face_recognizer = FaceRecognizer(
        model_path=face_recognizer_model_path, svm_model_path=svm_model_path)

    # open the source if not opened already
    if not source.isOpened():
        source.open()

    # initialise the sequence count
    sequence = 0

    # start the fps counter
    fps_counter.start()

    # loop through the frames in the video feed
    while True:
        ret, frame = source.read()
        frame = cv2.flip(frame, 1)

        # increment frame count; for fps calculation
        frame_count += 1

        # convert to grayscale
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # equalize the histogram
        grayImg = cv2.equalizeHist(grayImg)

        # detect the largest face
        face = face_detector.detect(grayImg)

        # if a face was detected
        if face is not None:

            # increment sequence count
            sequence += 1

            # get bounding boxes
            bb = clamp_rectangle(x1=face.left(), y1=face.top(
            ), x2=face.right(), y2=face.bottom(), x2_max=grayImg.shape[0]-1, y2_max=grayImg.shape[1]-1)

            # draw a rectangle around the detected face
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 255))

            # get the landmarks
            landmark = landmark_detector.predict(face, grayImg)

            # accumulate face and landmarks till we get a batch of batch_size
            if landmark is not None:
                faces.append(grayImg[bb[0]:bb[2], bb[1]:bb[3]])
                landmarks.append(landmark)

            # recognize the face in the batch
            if len(faces) == batch_size:
                face_embeddings = face_recognizer.embed(
                    images=faces, landmarks=landmarks)

                predicted_identity = face_recognizer.infer(
                    face_embeddings, threshold=face_recognition_confidence_threshold)

                print("Predicted identity:", predicted_identity)

                # start a new face recognition activity
                start_over()

        else:
            # start a new face recognition activity, because noo face face was detected in the frame
            start_over()

        # get frame rate
        fps_text = "FPS:" + str(fps)
        cv2.putText(frame, fps_text, (1, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow("output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cleanup()


def cleanup():
    """
    Runs cleanup services
    """
    global source
    global fps_counter
    global fps_queue

    source.release()
    cv2.destroyAllWindows()
    fps_counter.stop()

    print("Avg. FPS:", np.mean(np.array(fps_queue)))


def start_over():
    """
    Resets the following variables, if there is a discontinuity in detecting a face among consecutive frames:
    1. faces - all detected faces 
    2. landmarks - facial landmarks of the detected faces
    3. sequence - the counter for the sequence
    """

    global sequence
    global landmarks
    global faces

    sequence = 0
    faces = []
    landmarks = []


def fps_count():
    """
    Outputs the frames per second
    """
    global frame_count
    global fps
    global fps_list

    fps = frame_count
    fps_queue.append(fps)

    frame_count = 0


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        cleanup()
