from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import traceback

from model.deep.FaceRecognizer import FaceRecognizer
from process.FaceDetector import FaceDetector
from process.post_process.FaceAlign import FaceAlign
from process.post_process.LandmarkDetector import LandmarkDetector
from utils.set_interval import RepeatedTimer


source = None
fps_counter = None
frame_count = 0
fps = 0
sequence = 0
landmarks = []
faces = []


def run():

    global frame_count
    global source
    global fps_counter
    global fps
    global sequence
    global landmarks
    global faces

    face_area_threshold = 0.15
    camera_index = 0
    width, height = 150, 150
    batch_size = 32

    face_landmark_predictor_path = "pre_trained/shape_predictor_5_face_landmarks.dat"
    face_recognizer_model_path = "pre_trained/dlib_face_recognition_resnet_model_v1.dat"
    svm_model_path = "pre_trained/face_classifier.pkl"

    source = cv2.VideoCapture(index=camera_index)
    fps_counter = RepeatedTimer(interval=1.0, function=fps_count)

    face_detector = FaceDetector(face_area_threshold=face_area_threshold)
    landmark_detector = LandmarkDetector(
        predictor_path=face_landmark_predictor_path)
    face_aligner = FaceAlign(
        final_height=150, final_width=150, left_eye_offset=(0.25, 0.25))
    face_recognizer = FaceRecognizer(
        model_path=face_recognizer_model_path, svm_model_path=svm_model_path)

    if not source.isOpened():
        source.open()

    sequence = 0
    skip_factor = 100000

    fps_counter.start()

    while True:
        ret, frame = source.read()
        frame = cv2.flip(frame, 1)

        frame_count += 1

        # convert to grayscale
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # equalize the histogram
        grayImg = cv2.equalizeHist(grayImg)

        # detect the largest face
        face = face_detector.detect(grayImg)

        if face:
            sequence += 1

            bb = (face.left(), face.top(), face.right(), face.bottom())
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 255))
            print("Sequence:", sequence)

            # resize the face
            # face = cv2.resize(grayImg, (width, height),
            #                   interpolation=cv2.INTER_AREA)

            # get the landmarks
            landmark = landmark_detector.predict(face, grayImg)

            # add face and landmarks till we get a batch of batch_size
            faces.append(grayImg[bb[0]:bb[2], bb[1]:bb[3]])
            landmarks.append(landmark)

            if sequence == batch_size:
                face_embeddings = face_recognizer.embed(
                    images=faces, landmarks=landmarks)

                predicted_identity = face_recognizer.infer(face_embeddings)

                # cv2.putText(frame, fps_text, (bb[0], bb[1]-5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))

                print("predicted identity:", predicted_identity)

                start_over()

        else:
            start_over()

        # if sequence % skip_factor == 0:
        #     time.sleep(0.5)

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

    source.release()
    cv2.destroyAllWindows()
    fps_counter.stop()


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

    fps = frame_count
    #print("FPS:", fps)

    frame_count = 0


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        cleanup()
