import argparse
import sys
from pathlib import Path

import cv2 as cv
from matplotlib import pyplot as plt

from utils.general import colorstr, check_requirements

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path


def run(img1='',
        img2=''
        ):
    print(img1,img2)
    queryImage=cv.imread(img1,0)
    trainingImage=cv.imread(img2,0)#读取要匹配的灰度照片
    sift=cv.SIFT_create()#创建sift检测器
    kp1, des1 = sift.detectAndCompute(queryImage,None)
    kp2, des2 = sift.detectAndCompute(trainingImage,None)
    #设置Flannde参数
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams= dict(checks=50)
    flann=cv.FlannBasedMatcher(indexParams,searchParams)
    matches=flann.knnMatch(des1,des2,k=2)
    #设置好初始匹配值
    matchesMask=[[0,0] for i in range (len(matches))]
    for i, (m,n) in enumerate(matches):
        if m.distance< 0.7*n.distance: #舍弃小于0.7的匹配结果
            matchesMask[i]=[1,0]

    drawParams=dict(matchColor=(0,0,255),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0) #给特征点和匹配的线定义颜色
    resultimage=cv.drawMatchesKnn(queryImage,kp1,trainingImage,kp2,matches,None,**drawParams) #画出匹配的结果
    plt.imshow(resultimage,),plt.show()
    plt.imsave("test.jpg", resultimage)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, default='data/images/vis01.jpg')
    parser.add_argument('--img2', type=str, default='data/images/vis02.jpg')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('image matching: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)