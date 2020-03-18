from utils.utils import multi_gray_measure
import cv2
import os
import numpy as np

def mk(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("There are %d files in %s" % (len(os.listdir(path)), path))


if __name__ == "__main__":
    # img = cv2.imread(r"D:\GitHub\yolov3-point\data\images\train2014\4.jpg")
    # img = multi_gray_measure(img)
    # cv2.imshow("outwindow", img)
    # cv2.waitKey(0)
    # cv2.imwrite("outputsss.jpg", img)

    in_jpg_dir = "/home/dongpeijie/datasets/dimtargetSingle/images/data_sum"
    #r"D:\GitHub\yolov3-point\data\images\val2014"
    out_jpg_dir = "/home/dongpeijie/datasets/dimtargetSingle/images/saliency_train"
    #r"D:\GitHub\yolov3-point\data\images\valout"


    mk(out_jpg_dir)

    for i in os.listdir(in_jpg_dir):
        full_jpg_path = os.path.join(in_jpg_dir, i)

        print(full_jpg_path)
        img = cv2.imread(full_jpg_path)

        if img.all() == None:
            print("error: %s" % full_jpg_path)
        else:
            newimg = multi_gray_measure(img)
            newimg = newimg[...,np.newaxis]
            outimg = np.repeat(newimg, 3,
                               axis=2)  #img + np.repeat(newimg, 3, axis=2)

            outimg = np.clip(outimg, 0, 255)

            cv2.imwrite(os.path.join(out_jpg_dir,i), outimg)
