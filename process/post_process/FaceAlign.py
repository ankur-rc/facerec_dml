import dlib
from imutils import face_utils
import cv2
from LandmarkDetector import LandmarkDetector
import numpy as np
import imutils
import os
import shutil

"""
Inspired from 'Face Alignment with OpenCV and Python'
https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
"""

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


if __name__ == '__main__':

    dataset_path = '../../../../datasets/standard_att'
    #no_face_file = "no_face.txt"
    #multi_face_file = "multi_face.txt"

    normalized_dataset = "norm_" + \
        os.path.basename(os.path.realpath(dataset_path))

    if os.path.exists(normalized_dataset):
        shutil.rmtree(normalized_dataset)

    os.mkdir(normalized_dataset)

    detector = dlib.get_frontal_face_detector()
    predictor = LandmarkDetector()

    for root, dirs, files in os.walk(dataset_path):

        #print "dirs", dirs
        for direc in dirs:
            print "direc", direc
            extension = ".png"
            path = dataset_path + os.sep + direc + os.sep + '%d' + extension
            sequence = cv2.VideoCapture(path)

            while sequence.isOpened():
                ret, rgbImg = sequence.read()
                grayImg = None

                if rgbImg is None:
                    break
                elif rgbImg.shape[2] == 3:
                    grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
                else:
                    grayImg = rgbImg

                # equalize histogram
                grayImg = cv2.equalizeHist(grayImg)

                # find a face
                faces = detector(grayImg, 0)

                if len(faces) == 1:

                    # if len(faces) > 1:
                        # with open(multi_face_file, 'a') as f:
                        #     line = direc + ", " + \
                        #         str(int(sequence.get(cv2.CAP_PROP_POS_FRAMES)))+"\n"
                        #     f.write(line)

                        # get face (only interested if there's one and only one)
                    face = faces[0]

                    # get the landmarks
                    landmarks = predictor.predict(face, grayImg)

                    # get FaceAlign object
                    face_align = FaceAlign(
                        final_height=150, final_width=150, left_eye_offset=(0.25, 0.25))

                    # align the face
                    aligned_face = face_align.align(grayImg, landmarks)

                    # (x, y, w, h) = face_utils.rect_to_bb(face)
                    # original_face = imutils.resize(
                    #     rgbImg[y:y + h, x:x + w], width=256)

                    # display the output images
                    # cv2.imshow("Original", original_face)
                    # cv2.imshow("Aligned", aligned_face)

                    # cv2.circle(rgbImg, leye, 2, (0, 0, 255), -1)
                    # cv2.circle(rgbImg, reye, 2, (0, 0, 255), -1)

                    # cv2.line(rgbImg, reye, leye, (0, 0, 255))

                    # for (x, y) in landmarks:
                    #     cv2.circle(rgbImg, (x, y), 2, (0, 255, 0), -1)

                    # cv2.imshow("output", rgbImg)

                    # if cv2.waitKey(500) & 0xFF == ord('q'):
                    #     sequence.release()
                    #     cv2.destroyAllWindows()
                    #     quit()

                    if not os.path.exists(normalized_dataset+os.sep+direc):
                        os.mkdir(normalized_dataset+os.sep+direc)

                    filename = normalized_dataset+os.sep+direc + os.sep + \
                        str(int(sequence.get(cv2.CAP_PROP_POS_FRAMES))) + ".pgm"
                    cv2.imwrite(filename, aligned_face)

                else:
                    pass
                    # with open(no_face_file, 'a') as f:
                    #     line = direc + ", " + \
                    #         str(int(sequence.get(cv2.CAP_PROP_POS_FRAMES)))+"\n"
                    #     f.write(line)
