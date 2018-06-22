import dlib
import cv2
from imutils import face_utils

if __name__ == "__main__":

    dataset_path = '../../../../datasets/cyber_extruder_ultimate'
    image = '/000008/000007.jpg'

    img = cv2.imread(dataset_path+image)
    detector = dlib.get_frontal_face_detector()

    faces = detector(img)

    for face in faces:
        cv2.rectangle(img, (face.left(), face.top()),
                      (face.right(), face.bottom()), (0, 0, 255))

    cv2.imshow("output", img)

    cv2.waitKey()
