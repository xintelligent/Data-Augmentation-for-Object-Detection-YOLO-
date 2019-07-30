import configparser
import glob
import os

import cv2 as cv
import numpy as np

from SampleImgInterface import SampImgModifier

DEFAULT_PARAMS = {
'BackgroundFilePath': './MyData/background',
'SampleFilesPath': './MyData/SuperXping/img',
'bgColor': 255,
'bgTthresh': 8,
'maxXangle': 50,
'maxYangle': 50,
'maxZangle': 50,
'maxAngle_Affine': 30,
'outputPerSample': 100,
'GausNoiseProb': 0.2,
'MedianNoiseProb': 0.1,
'AffineRotateProb': 0.3,
'SharpenProb': 0.2,
'PerspTransProb': 0.8,
'ScalingProb': 0.7,
'BrightnessProb': 1,
'OutputPath': './TransferData/SuperXping'
}


def placeDistortedSample(outImgTight, foregroundPixTight, BoundRect, bkgImg):

    bgHeight, bgWidth, _ = np.shape(bkgImg)
    outHeight, outWidth, _ = np.shape(outImgTight)

    if outHeight < bgHeight and outWidth < bgWidth:

        finalImg = np.array(bkgImg).copy()

        posX = np.random.randint(0, bgWidth - outWidth)
        if posX + outWidth > bgWidth:
            posX = bgWidth - outWidth - 10

        posY = np.random.randint(0, bgHeight-10)
        if posY + outHeight > bgHeight - outHeight:
            posY = bgHeight - outHeight - 10

        indices = np.zeros((np.shape(foregroundPixTight)), np.uint64)
        indices[0] = np.array([foregroundPixTight[0]]) + posY
        indices[1] = np.array([foregroundPixTight[1]]) + posX

        boundRectFin = np.zeros((2, 2), float)
        # The order of x and y have been reversed for yolo
        boundRectFin[1][1] = float(BoundRect[1][0] - BoundRect[0][0]) / float(bgHeight)
        boundRectFin[1][0] = float(BoundRect[1][1] - BoundRect[0][1]) / float(bgWidth)
        boundRectFin[0][1] = float(posY) / float(bgHeight)+boundRectFin[1][1] / float(2)
        boundRectFin[0][0] = float(posX) / float(bgWidth)+boundRectFin[1][0] / float(2)

        foregroundpixBkg = tuple(map(tuple, indices))
        finalImg[foregroundpixBkg] = outImgTight[foregroundPixTight]
        return True, finalImg, boundRectFin
    else:
        return False, 0, 0


def main():

    parser = configparser.RawConfigParser(defaults=DEFAULT_PARAMS)
    parser.read('MyParameters.config')

    samplePath = parser.get('MY_PARAMS', 'sampleFilesPath')
    bgColor = int(parser.get('MY_PARAMS', 'bgColor'))
    bgThresh = int(parser.get('MY_PARAMS', 'bgThresh'))
    maxXangle_Persp = int(parser.get('MY_PARAMS', 'maxXangle'))
    maxYangle_Persp = int(parser.get('MY_PARAMS', 'maxYangle'))
    maxZangle_Persp = int(parser.get('MY_PARAMS', 'maxZangle'))
    maxAngle_Affine = int(parser.get('MY_PARAMS', 'maxAngle_Affine'))
    GaussianNoiseProb = float(parser.get('MY_PARAMS', 'GausNoiseProb'))
    MedianNoiseProb = float(parser.get('MY_PARAMS', 'MedianNoiseProb'))
    SharpenProb = float(parser.get('MY_PARAMS', 'SharpenProb'))
    PerspTransProb = float(parser.get('MY_PARAMS', 'PerspTransProb'))
    ScalingProb = float(parser.get('MY_PARAMS', 'ScalingProb'))
    BrightnesProb = float(parser.get('MY_PARAMS', 'BrightnessProb'))
    outputPerSample = float(parser.get('MY_PARAMS', 'outputPerSample'))
    AffineRotateProb = float(parser.get('MY_PARAMS', 'AffineRotateProb'))

    image_count = 0
    for sampleImgPath in glob.glob(os.path.join(samplePath, '*_area.JPG')):
        sampleImg = cv.imread(sampleImgPath)
        dimensions = np.shape(sampleImg)
        image_count += 1
        print("正在处理: ", sampleImgPath, image_count)
        lower = np.array([bgColor - bgThresh, bgColor - bgThresh, bgColor - bgThresh])
        upper = np.array([bgColor + bgThresh, bgColor + bgThresh, bgColor + bgThresh])
        ImgModifier = SampImgModifier(sampleImg, dimensions, lower, upper, bgColor)
        GaussianNoiseFlag  = np.less(np.random.uniform(0, 1), GaussianNoiseProb)
        MedianNoiseFlag    = np.less(np.random.uniform(0, 1), MedianNoiseProb)
        SharpenFlag        = np.less(np.random.uniform(0, 1), SharpenProb)
        PersTransFlag      = np.less(np.random.uniform(0, 1), PerspTransProb)
        ScalingFlag        = np.less(np.random.uniform(0, 1), ScalingProb)
        BrightnessFlag     = np.less(np.random.uniform(0, 1), BrightnesProb)
        AffineRotateFlag   = np.less(np.random.uniform(0, 1), AffineRotateProb)

        PersTransFlag = False

        if PersTransFlag:
            ImgModifier.perspectiveTransform(maxXangle_Persp, maxYangle_Persp, maxZangle_Persp, bgColor)

        if AffineRotateFlag and not PersTransFlag:
            ImgModifier.affineRotate(maxAngle_Affine, bgColor)

        if GaussianNoiseFlag:
            ImgModifier.addGaussianNoise(0, 5)

        if MedianNoiseFlag and not GaussianNoiseFlag:
            percentPixels = 0.2
            percentSalt = 0.3
            ImgModifier.addMedianNoise(percentPixels, percentSalt)

        if SharpenFlag and not MedianNoiseFlag and not GaussianNoiseFlag:
            ImgModifier.sharpenImage()

        if ScalingFlag:
            scale = np.random.uniform(0.4, 0.8)
            ImgModifier.scaleImage(scale)

        if BrightnessFlag and not SharpenFlag and not MedianNoiseFlag and not GaussianNoiseFlag:
            scale = np.random.uniform(0.5, 1)
            ImgModifier.modifybrightness(scale)

        # cv2.imwrite('messigray.png', img)

        # 保存变形后的图片
        koutu_transfer_filename = sampleImgPath[sampleImgPath.rfind("/"):].replace("/", str(image_count) + "_")
        cv.imwrite("./koutu/SuperXping/" + koutu_transfer_filename, ImgModifier.modifiedImg)


if __name__ == '__main__':
    main()