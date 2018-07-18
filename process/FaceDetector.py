from __future__ import division
import cv2
import dlib
import numpy
from imutils import face_utils


class FaceDetector():

    def __init__(self, face_area_threshold=0.25):
        """
        Initialise a 'FaceDetector' object

        :param face_area_threshold: minimum area the face must cover w.r.t the frame
        :type largest_only: float [0,1]
        """

        self.detector = dlib.get_frontal_face_detector()
        self.face_area_threshold = face_area_threshold

    def detect(self, gray_frame):
        """
        Detect faces in the frame

        :param gray_frame: grayscale image that might include a face
        :type gray_frame: numpy.ndarray
        :return bounding_box: bounding box coordinates signifying the location of the face
        :rtype bounding_box: dlib.rectangle
        """

        bounding_box = None

        faces = self.detector(gray_frame, 0)
        areas = [0 for face in faces]
        frame_area = gray_frame.shape[0]*gray_frame.shape[1]
        # print 'frame area is (', gray_frame.shape, "):", frame_area

        if len(faces) > 0:
            for idx, face in enumerate(faces):
                h, w = face.height(), face.width()
                # print "Face ", idx, ":", h, w, ":", h*w
                areas[idx] = h*w

            largest_face_idx = numpy.argmax(numpy.array(areas))

            # print "Areas are:", areas
            # print "Largest face area index:", largest_face_idx
            # print "Face area ratio:", areas[largest_face_idx]/frame_area

            if areas[largest_face_idx]/frame_area > self.face_area_threshold:
                bounding_box = faces[largest_face_idx]

        return bounding_box


if __name__ == "__main__":

    detector = FaceDetector(face_area_threshold=0.001)
    file = "groupie.jpg"

    frame = cv2.imread(file)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    face = detector.detect(gray)

    if face:
        bb = (face.left(), face.top(), face.right(), face.bottom())
        cv2.rectangle(gray, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 255))
        # cv2.putText(gray, str(idx), (bb[0]-3, bb[1]-3),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("op", gray)

    if cv2.waitKey():
        cv2.destroyAllWindows()
