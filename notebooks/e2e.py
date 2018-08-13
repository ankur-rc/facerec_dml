from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import thread
import logging
import tqdm
import pprint

import face_trigger

from face_trigger.model.deep.FaceRecognizer import FaceRecognizer
from face_trigger.process.post_process import FaceDetector, LandmarkDetector, FaceAlign
from face_trigger.utils.common import RepeatedTimer, clamp_rectangle
from face_trigger.utils.Dataset import Dataset

import uuid


def normalize_dataset(dataset_path=None, output_path=None):

    logger = logging.getLogger(__name__)

    face_detector = FaceDetector()
    face_align = FaceAlign(left_eye_offset=(0.35, 0.35), final_width=150)
    landmark_predictor = LandmarkDetector()

    rejected_faces = {}

    bar = tqdm.tqdm(total=None)

    if not os.path.exists(dataset_path):
        raise Exception("Invalid dataset path!")

    # setup output directory
    if os.path.isdir(output_path):
        os.rename(output_path, os.path.join(os.path.split(
            output_path)[0], os.path.split(
            output_path)[1] + "_" + str(uuid.uuid4().get_hex())))
    os.makedirs(output_path)

    for root, dirs, files in os.walk(dataset_path):

        if root == dataset_path:
            bar.total = len(dirs)

        for direc in dirs:
            # create output directory for this presonality
            output_direc_path = os.path.join(output_path, direc)
            os.mkdir(output_direc_path)

        for img in files:

            img_path = os.path.join(dataset_path, root, img)

            # read the image
            rgbImg = cv2.imread(img_path)

            grayImg = None
            if rgbImg is None:
                break
            elif rgbImg.shape[2] == 3:
                grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
            else:
                grayImg = rgbImg

            # detect faces
            faces = face_detector.detect_unbounded(grayImg)

            if len(faces) == 1:

                # get face (only interested if there's one and only one)
                face_bb = faces[0]

                # get the landmarks
                landmarks = landmark_predictor.predict(face_bb, grayImg)

                # align the face
                aligned_face = face_align.align(grayImg, landmarks)

                # write to output directory
                save_path = os.path.join(
                    output_path, os.path.basename(root), img)

                cv2.imwrite(save_path, aligned_face)

            else:
                root = os.path.basename(root)
                if root in rejected_faces:
                    rejected_faces[root].append(img)
                else:
                    rejected_faces[root] = [img]

        if root != dataset_path:
            bar.update()

    bar.close()
    logger.info("Normalized dataset created at {}".format(output_path))

    print("Rejected directories:")
    pprint.pprint(rejected_faces)

    return rejected_faces


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset_path = "/media/ankurrc/new_volume/softura/facerec/datasets/standard_att_copy"
    output_path = "/media/ankurrc/new_volume/softura/facerec/att_norm"

    normalize_dataset(
        dataset_path=dataset_path, output_path=output_path)
