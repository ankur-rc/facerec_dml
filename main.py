import cv2
from model.deep.FaceRecognizer import FaceRecognizer
from process.FaceDetector import FaceDetector
from process.post_process.FaceAlign import FaceAlign
from process.post_process.LandmarkDetector import LandmarkDetector


def run():

    source = cv2.VideoCapture(index=0)
    face_detector = FaceDetector(face_area_threshold=0.15)

    if not source.isOpened():
        source.open()

    sequence = 0

    while True:
        ret, frame = source.read()
        frame = cv2.flip(frame, 1)

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # equalize the histogram
        gray = cv2.equalizeHist(gray)

        # detect the largest face
        face = face_detector.detect(gray)

        if face:
            bb = (face.left(), face.top(), face.right(), face.bottom())
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 255))
            sequence += 1
            print "Sequence:", sequence

        else:
            sequence = 0

        cv2.imshow("output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
