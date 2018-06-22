from imutils import face_utils
import dlib
import cv2
import os
from time import sleep


class LandmarkDetector:

    """
        A landmark detector that annotates face bounding boxes with 5 landmarks
    """

    def __init__(self, predictor_path="shape_predictor_5_face_landmarks.dat"):
        """
        Instantiates the 'LandmarkDetector' object

        :param predictor_path: path to trained face predictor model
        :type predictor_path: str
        """
        self.predictor = dlib.shape_predictor(predictor_path)

    def predict(self, bounding_box, grayImg):
        """
        Provides an array of tuples for facial landmarks, predicted within a bounding box

        :param bounding_box: bounding box coordinates in dlib format
        :type bounding_box: dlib.rectangle
        :param grayImg: grayscale image
        :type grayImg: numpy.ndarray
        :return landmarks: 5-tuple
        :rtype landmarks: numpy.ndarray
        """

        shape = self.predictor(grayImg, bounding_box)
        shape = face_utils.shape_to_np(shape)

        return shape


if __name__ == '__main__':

    trained_model = "shape_predictor_5_face_landmarks.dat"
    dataset_path = '../../../../datasets/cyber_extruder_ultimate'

    detector = dlib.get_frontal_face_detector()
    predictor = LandmarkDetector(trained_model)

    for root, dirs, files in os.walk(dataset_path):
        for direc in dirs:
            path = dataset_path + os.sep + direc + os.sep + '00000%d.jpg'
            sequence = cv2.VideoCapture(path)
            while True:
                ret, rgbImg = sequence.read()
                grayImg = None

                if rgbImg is None:
                    break
                elif rgbImg.shape[2] == 3:
                    grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
                else:
                    grayImg = rgbImg

                bounding_boxes = detector(grayImg, 0)

                for (i, bounding_box) in enumerate(bounding_boxes):

                    shape = predictor.predict(bounding_box, grayImg)

                    for (x, y) in shape:
                        cv2.circle(rgbImg, (x, y), 2, (0, 255, 0), -1)

                cv2.imshow("Output", rgbImg)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    sequence.release()
                    cv2.destroyAllWindows()
                    quit()
