import dlib
from imutils import face_utils
import cv2
from LandmarkDetector import LandmarkDetector
import numpy as np
import imutils
import os
import shutil

FACE_LANDMARK_INDICES = {
    "reye_right": 0,
    "reye_left": 1,
    "leye_right": 3,
    "leye_left": 2,
    "nose": 4
}


class FaceAlign():
    """
    Align a face by performing affine transformations.
    """

    def __init__(self, left_eye_offset=(0.35, 0.35), final_width=256, final_height=None):
        """
        Instantiate a 'FaceAlign' object

        :param left_eye_offset: the amount by which to anchor the left eye center
        :type left_eye_offset: 2-tuple
        :param final_width: the width in pixels of the aligned image
        :type final_width: int
        :param final_height: the height in pixels of the aligned image
        :type final_height: int
        """

        self.left_eye_offset = left_eye_offset
        self.final_width = final_width

        if final_height is None:
            self.final_height = final_width
        else:
            self.final_height = final_height

    def align(self, img, landmarks):
        """
        Align the given image according to given face landmarks -
        1. Eyes are aligned on a horizontal axis
        2. Face is scaled to keep eye centers on the same offset location w.r.t the face
        3. Face is centered on the center of mass of the eyes.

        :param img: rgb or gray-scale image/frame
        :type img: numpy.ndarray
        :landmarks: the x,y-coordinates of the facial landmarks
        :type landmarks: dlib shape
        :return aligned_img: aligned image of final dimensions
        :rtype: numpy.ndarray
        """

        landmarks = face_utils.shape_to_np(landmarks)

        # eye centers
        leye = tuple(np.add(landmarks[FACE_LANDMARK_INDICES["leye_right"]],
                            landmarks[FACE_LANDMARK_INDICES["leye_left"]])/2)

        reye = tuple(np.add(landmarks[FACE_LANDMARK_INDICES["reye_right"]],
                            landmarks[FACE_LANDMARK_INDICES["reye_left"]])/2)

        # angle between eye centers
        dY = reye[1] - leye[1]
        dX = reye[0] - leye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # right eye coordinates as a function of the left eye
        right_eye_offset_x = 1.0 - self.left_eye_offset[0]

        # scale
        current_dist = np.sqrt(dY**2+dX**2)
        desired_dist = (right_eye_offset_x -
                        self.left_eye_offset[0])*self.final_width
        scale = desired_dist/current_dist

        # median of eye centers
        eye_median = ((reye[0]+leye[0])//2, (reye[1] + leye[1])//2)

        # transformation matrix
        transform_matrix = cv2.getRotationMatrix2D(eye_median, angle, scale)

        # update the matrix's transaltion compopnents
        tX = self.final_width * 0.5
        tY = self.final_height * self.left_eye_offset[1]
        transform_matrix[0, 2] += (tX - eye_median[0])
        transform_matrix[1, 2] += (tY - eye_median[1])

        # align the face
        aligned_face = cv2.warpAffine(
            img, transform_matrix, (self.final_width, self.final_height), flags=cv2.INTER_CUBIC)

        return aligned_face
