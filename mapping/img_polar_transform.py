import cv2
import numpy as np

view_angle = 52

scene = ["Beechwood_0_int"]

DS_num = 16
DS_length = 400

for cnt in range(DS_length):
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS

    imagepath = "./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_" + str(DS_num) + "/seg_trapezoid/DS" + str(DS_num) + "_seg_" + str(cnt) + ".jpg"
    Input = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

    src = np.rot90(Input)
    src = np.ascontiguousarray(src)
    if src is None:
        print("Could not initialize capturing...\n")

    center = (int(src.shape[1]), int(src.shape[0] / 2))
    maxRadius = Input.shape[0]

    # direct transform
    lin_polar_img = cv2.warpPolar(src, None, center, maxRadius, flags)

    for i in range(lin_polar_img.shape[0]):
        for j in range(5, lin_polar_img.shape[1]):
            if lin_polar_img[i, j] <= 20:
                lin_polar_img[i, j:] = 0
                break

    lin_polar_img = lin_polar_img[int((lin_polar_img.shape[0]/2-view_angle)) :
                                  -int((lin_polar_img.shape[0]/2-view_angle))]
    lin_polar_img = cv2.resize(lin_polar_img, (224, 224))

    lin_polar_img = cv2.rotate(lin_polar_img, cv2.ROTATE_90_CLOCKWISE)  # rotation 90 degree
    lin_polar_img = cv2.rotate(lin_polar_img, cv2.ROTATE_90_CLOCKWISE)
    lin_polar_img = cv2.rotate(lin_polar_img, cv2.ROTATE_90_CLOCKWISE)

    _, lin_polar_img = cv2.threshold(lin_polar_img, 20, 255, cv2.THRESH_BINARY)

    cv2.imwrite("./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_" + str(DS_num) + "/seg_scallop/DS" + str(DS_num) + "_seg_" + str(cnt) + ".jpg",
                lin_polar_img)



    lin_polar_img_padding = np.zeros((704, 224))
    lin_polar_img_padding[int(704/2-view_angle):int(704/2+view_angle)] = cv2.resize(cv2.rotate(lin_polar_img, cv2.ROTATE_90_CLOCKWISE),
                                                      (224, view_angle*2))

    # inverse transform
    recovered_lin_polar_img = cv2.warpPolar(lin_polar_img_padding, (src.shape[0], src.shape[1] * 2), center, maxRadius,
                                            flags | cv2.WARP_INVERSE_MAP)

    recovered_lin_polar_img = recovered_lin_polar_img[:224]
    recovered_lin_polar_img = cv2.rotate(recovered_lin_polar_img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imwrite("./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_" + str(DS_num) + "/seg_scallop/seg_scallop_origin/DS" + str(DS_num) + "_seg_" + str(cnt) + ".jpg",
                recovered_lin_polar_img)

