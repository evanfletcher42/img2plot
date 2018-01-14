import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.misc
import numpy as np
import math
import skimage.exposure
import skimage.draw
import svgwrite

# ---------------------------------------------------------------------------------------------------------------------
# Input and Output

INPUT_IMAGE_PATH = "path/to/input/file.png"

OUTPUT_SVG_PATH = "path/to/results/file.svg"

# ---------------------------------------------------------------------------------------------------------------------
# Configuration parameters.  These affect the appearance of the output image.

# Program will continue drawing lines on the highest-intensity edges until the max intensity value drops below
# this fraction of the initial peak value. A smaller number means more lines will be drawn.
TERMINATION_RATIO = 1.0/3.5

# A line is extended until the edge intensity drops below this fraction of the corresponding peak edge intensity.
# Larger values mean many small lines; smaller values cause lines to extend for longer distances.
LINE_CONTINUE_THRESH = 0.01

# Lines must be longer than this length in pixels, else they will not be drawn.
# Smaller numbers can mean lots of short lines, but will represent small details more faithfully.
# Larger numbers can result in an aesthetically-pleasing "sketchy" look - losing small details.
MIN_LINE_LENGTH = 21

# Sets the amount of angle change before a line will be terminated.
# Small numbers mean many short lines; large numbers mean longer lines, but they cut corners.
MAX_CURVE_ANGLE_DEG = 20.0

# When drawing / extending lines, each new pixel contributes to the line direction, via a low-pass filter with this
# attack value.  Higher numbers mean lines can turn faster.
LPF_ATK = 0.05

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the monochrome image as a preprocessing step.
# This can help bring out more details, especially if the input image has large areas of bright and dark.
USE_CLAHE = True
CLAHE_KERNEL_SIZE = 32

# Apply a Gaussian blur as a preprocessing step.
# This can help generate longer lines if the input image is noisy or otherwise high-frequency.
USE_GAUSSIAN_BLUR = True
GAUSSIAN_KERNEL_SIZE = 1

# ---------------------------------------------------------------------------------------------------------------------
# Utility functions

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def bilinearInterpolate(img, (x,y)):
    xfloat = x - math.floor(x)
    yfloat = y - math.floor(y)

    xfloor = math.floor(x)
    yfloor = math.floor(y)

    xceil = math.ceil(x)
    yceil = math.ceil(y)

    if(xfloor < 0):
        xfloor = 0
    if(xceil >= img.shape[1]):
        xceil = img.shape[1]-1

    if (yfloor < 0):
        yfloor = 0
    if (yceil >= img.shape[0]):
        yceil = img.shape[0] - 1

    topLeft = img[int(yfloor), int(xfloor)]
    topRight = img[int(yfloor), int(xceil)]
    bottomLeft = img[int(yceil), int(xfloor)]
    bottomRight = img[int(yceil), int(xceil)]

    topMid = xfloat * topRight + (1-xfloat) * topLeft
    botMid = xfloat * bottomRight + (1-xfloat) * bottomLeft

    mid = yfloat * botMid + (1-yfloat) * topMid

    return mid

def getLineFromGradient(img, (px, py), (gradx, grady)):
    angle = math.atan2(grady[py,px], gradx[py,px])

    # Attempt to grow the line as much as possible.
    # The line can continue forever at the given angle, until:
    #   - the Sobel value changes too much
    #   - the gradient direction changes too much (more than 5 degrees)

    len_left = 0
    len_right = 0

    startx = px
    starty = py
    endx = px
    endy = py

    mangle = angle

    # grow the "start" side (calling it "left" because reasons)
    while 0 < starty < img.shape[0]-1 and 0 < startx < img.shape[1] - 1 \
            and bilinearInterpolate(img, (startx, starty)) > LINE_CONTINUE_THRESH * img[py, px]:

        len_left += 1

        # Recalculate angle.  This allows lines to "follow" curves, at least a little.
        cangle = math.atan2(grady[int(round(starty)), int(round(startx))],
                            gradx[int(round(starty)), int(round(startx))])

        # low pass filtered angle update
        mangle = mangle * (1-LPF_ATK) + cangle * LPF_ATK

        if abs(angle - mangle) > MAX_CURVE_ANGLE_DEG * (2 * math.pi / 360):
            break

        startx = pIdxCol + len_left * math.sin(mangle)
        starty = pIdxRow - len_left * math.cos(mangle)

    mangle = angle

    # grow the "end" side (calling it "right" because reasons)
    while 0 < endy < img.shape[0]-1 and 0 < endx < img.shape[1] - 1 \
            and bilinearInterpolate(img, (endx, endy)) > LINE_CONTINUE_THRESH * img[py, px]:

        len_right += 1

        # Recalculate angle.  This allows lines to "follow" curves, at least a little.
        cangle = math.atan2(grady[int(round(endy)), int(round(endx))],
                            gradx[int(round(endy)), int(round(endx))])

        # low pass filtered angle update
        mangle = mangle * (1 - LPF_ATK) + cangle * LPF_ATK

        if abs(angle - mangle) > MAX_CURVE_ANGLE_DEG * (2 * math.pi / 360):
            break

        endx = pIdxCol - len_right * math.sin(mangle)
        endy = pIdxRow + len_right * math.cos(mangle)

    return int(round(startx)), int(round(starty)), int(round(endx)), int(round(endy)), (len_left + len_right + 1)

dwg = svgwrite.Drawing(OUTPUT_SVG_PATH, profile='tiny')

baseImage = scipy.misc.imread(INPUT_IMAGE_PATH)
baseImageGray = rgb2gray(baseImage)

# normalize 0..1
normImgGray = baseImageGray - baseImageGray.min()
normImgGray = normImgGray / normImgGray.max()

# CLAHE (brings out details)
if USE_CLAHE:
    normImgGray = skimage.exposure.equalize_adapthist(normImgGray, kernel_size=CLAHE_KERNEL_SIZE)

# Gaussian blur (gets rid of details)
if USE_GAUSSIAN_BLUR:
    normImgGray = scipy.ndimage.filters.gaussian_filter(normImgGray, GAUSSIAN_KERNEL_SIZE)

plt.imshow(normImgGray, cmap='gray')
plt.show()

sobelDx = ndimage.sobel(normImgGray, 0)  # horizontal
sobelDy = ndimage.sobel(normImgGray, 1)  # vertical
mag = np.hypot(sobelDx, sobelDy)

# where the image is locally darker in low frequencies, increase the probability of drawing a line.
imgBlur = scipy.ndimage.filters.gaussian_filter(normImgGray, 2)
mag = np.multiply(mag, imgBlur.max()-imgBlur)

# turn mag into a proper probability distribution function
mag = mag / np.sum(mag)

plt.imshow(mag)
plt.colorbar()
plt.show()

magGradY, magGradX = np.gradient(normImgGray)

plt.imshow(magGradX)
plt.show()

plt.imshow(magGradY)
plt.show()

lineImg = np.zeros(mag.shape)
lineImg = lineImg - 1

anglehist = []
linehist = []
# Sample a lot of edges

outImg = np.zeros(mag.shape, dtype=np.uint8)

initmaxp = mag.max()
cmax = initmaxp
i = 0

llacc = 0.0
llcnt = 0.0
minll = 1e9
maxll = 0

while cmax > initmaxp*TERMINATION_RATIO:
    i = i+1
    if i % 250 == 0:
        print "Max P: ", mag.max(), " term at:", initmaxp*TERMINATION_RATIO
        print "Line Stats: N=", llcnt, "length: min", minll, "mean", llacc/llcnt, "max", maxll
        llacc = 0
        llcnt = 0
        minll = 99999
        maxll = 0

    pixIdx = mag.argmax()

    pIdxRow = int(pixIdx / mag.shape[1])
    pIdxCol = pixIdx % mag.shape[1]

    cmax = mag[pIdxRow, pIdxCol]
    # print pIdxRow, ",", pIdxCol

    (lstartx, lstarty, lendx, lendy, totalLength) = getLineFromGradient(mag, (pIdxCol, pIdxRow), (magGradX, magGradY))

    if totalLength < MIN_LINE_LENGTH:
        # this is too short, and is not worth drawing
        # We don't want to shorten other lines, so replace this peak with the (guaranteed smaller) mean of its neighbors
        acc = 0.0
        cnt = 0
        if pIdxRow+1 < mag.shape[0]:
            acc = acc + mag[pIdxRow+1, pIdxCol]
            cnt = cnt+1

        if pIdxCol + 1 < mag.shape[1]:
            acc = acc + mag[pIdxRow, pIdxCol+1]
            cnt = cnt + 1

        if pIdxRow - 1 >= 0:
            acc = acc + mag[pIdxRow - 1, pIdxCol]
            cnt = cnt + 1

        if pIdxCol - 1 >= 0:
            acc = acc + mag[pIdxRow, pIdxCol - 1]
            cnt = cnt + 1

        mag[pIdxRow, pIdxCol] = acc / cnt

        continue

    # draw this line in the SVG image, too
    dwg.add(dwg.line((lstartx, lstarty), (lendx, lendy), stroke=svgwrite.rgb(0,0,0,'%')))

    # collect line statistics:

    # accumulator for mean & line count
    llacc = llacc + totalLength
    llcnt = llcnt + 1

    # min/max line lengths
    if totalLength < minll:
        minll = totalLength

    if totalLength > maxll:
        maxll = totalLength

    # draw the line on the edge image and preview images.

    rr, cc, val = skimage.draw.line_aa(lstarty, lstartx, lendy, lendx)
    rrd, ccd = skimage.draw.line(lstarty, lstartx, lendy, lendx)

    rr[rr < 0] = 0
    rr[rr >= mag.shape[0]] = mag.shape[0]-1

    cc[cc < 0] = 0
    cc[cc >= mag.shape[1]] = mag.shape[1] - 1

    rrd[rrd < 0] = 0
    rrd[rrd >= mag.shape[0]] = mag.shape[0] - 1

    ccd[ccd < 0] = 0
    ccd[ccd >= mag.shape[1]] = mag.shape[1] - 1

    # Draw the line in the preview image, for visualization purposes
    outImg[rrd, ccd] = 255

    # Draw the line in the edge magnitude image
    mag[rr, cc] = 0
    mag[pIdxRow, pIdxCol] = 0  # also knock down the peak intensity that created this line, because the lines can move

outImg[outImg > 255] = 255

outImg = -1*outImg + 255

dwg.save()

plt.imshow(outImg, cmap='gray')
plt.show()

