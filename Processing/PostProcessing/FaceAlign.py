import dlib
from imutils import face_utils
import cv2
from LandmarkDetector import LandmarkDetector
import numpy as np
import imutils

"""
https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
"""

LANDMARK_INDEX = {
    "reye_right": 0,
    "reye_left": 1,
    "leye_right": 3,
    "leye_left": 2,
    "nose": 4
}

if __name__ == '__main__':

    image = "../../../../datasets/cyber_extruder_ultimate/000024/000002.jpg"
    rgbImg = cv2.imread(image)
    grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)

    # find a face
    detector = dlib.get_frontal_face_detector()
    faces = detector(grayImg, 0)
    landmark_detector = LandmarkDetector()

    # get face (only interested if there's one and one only)
    face = faces[0]

    # get the landmarks
    landmarks = landmark_detector.predict(face, grayImg)
    # cv2.rectangle(rgbImg, (face.left(), face.top()),
    #               (face.right(), face.bottom()), (0, 0, 255))

    # eye centers
    leye = tuple(np.add(landmarks[LANDMARK_INDEX["leye_right"]],
                        landmarks[LANDMARK_INDEX["leye_left"]])/2)

    reye = tuple(np.add(landmarks[LANDMARK_INDEX["reye_right"]],
                        landmarks[LANDMARK_INDEX["reye_left"]])/2)

    # angle between eye centers
    dY = reye[1] - leye[1]
    dX = reye[0] - leye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # right eye coordinates as a function on left eye
    desired_leye = (0.35, 0.35)
    desired_reye_x = 1.0 - desired_leye[0]

    # scale
    desired_width = 256
    desired_height = desired_width
    current_dist = np.sqrt(dY**2+dX**2)
    desired_dist = (desired_reye_x - desired_leye[0])*desired_width
    scale = desired_dist/current_dist

    # median between eye centers
    eye_median = ((reye[0]+leye[0])//2, (reye[1] + leye[1])//2)

    transform_matrix = cv2.getRotationMatrix2D(eye_median, angle, scale)

    # update the matrix's transaltion compopnents
    tX = desired_width * 0.5
    tY = desired_height * desired_leye[1]
    transform_matrix[0, 2] += (tX - eye_median[0])
    transform_matrix[1, 2] += (tY - eye_median[1])

    # apply affine transformation

    aligned_face = cv2.warpAffine(
        rgbImg, transform_matrix, (desired_width, desired_height), flags=cv2.INTER_CUBIC)

    (x, y, w, h) = face_utils.rect_to_bb(face)
    original_face = imutils.resize(rgbImg[y:y + h, x:x + w], width=256)

    # display the output images
    cv2.imshow("Original", original_face)
    cv2.imshow("Aligned", aligned_face)

    # cv2.circle(rgbImg, leye, 2, (0, 0, 255), -1)
    # cv2.circle(rgbImg, reye, 2, (0, 0, 255), -1)

    # cv2.line(rgbImg, reye, leye, (0, 0, 255))

    # for (x, y) in landmarks:
    #     cv2.circle(rgbImg, (x, y), 2, (0, 255, 0), -1)

    #cv2.imshow("output", rgbImg)

    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()
