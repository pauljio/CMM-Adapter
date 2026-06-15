import numpy as np
from PIL import Image


if __name__ == "__main__":

    # np.set_printoptions(threshold=int(1e10))

    dis = np.array(Image.open('/data1/cjp/Datasets/NYU_Depth_V2/labels_train/0795.png'))
    print(dis.dtype)
    print(dis.shape)
    print(f"max: {np.max(dis)}")
    print(f"min: {np.min(dis)}")
    # print(dis)

    # dis[dis > 0] = (dis[dis > 0] - 1.0) / 256.0
    # print(dis.dtype)
    # print(dis[500:600, 500:600])
    #
    # disp = dis[270, 106]
    # print(disp)
    # depth = (0.209313 * 2262.52) / disp
    # print(depth)

