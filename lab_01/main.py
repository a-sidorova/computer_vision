import cv2
import os
from canny import Canny


def main():
    print("*==============* First lab *==============*")
    path, file = os.path.split(os.path.abspath(__file__))
    path_to_img = path + "\\..\\data\\lenna.png"
    print("Src Image: ", path_to_img)
    print("0. Read Image")
    image = cv2.imread(path_to_img)
    print("1. Face Detector")
    # pass
    print("2. Image indent")
    # pass
    print("3. Edge Detector")
    edges_img = Canny.get(image)
    cv2.imshow('Edge Detector', edges_img)
    cv2.waitKey()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

