"""img2plot: turn images into drawings for a pen plotter.

Usage:
    python img2plot.py input.png -o output.svg [--style curves|lines] [options]

Run with --help to see every tuning parameter.
"""

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import skimage.draw
import skimage.exposure
import skimage.io
import svgwrite

# ---------------------------------------------------------------------------------------------------------------------
# Configuration parameters.  These affect the appearance of the output image.  Each can be overridden on the command
# line; see --help.

@dataclass
class Params:
    # "curves" draws each stroke as a smooth Bezier path that bends to follow edges.
    # "lines" draws each stroke as a single straight segment (the original img2plot look).
    style: str = "curves"

    # Program will continue drawing lines on the highest-intensity edges until the max intensity value drops below
    # this fraction of the initial peak value. A smaller number means more lines will be drawn.
    termination_ratio: float = 1.0 / 3.5

    # A line is extended until the edge intensity drops below this fraction of the corresponding peak edge intensity.
    # Larger values mean many small lines; smaller values cause lines to extend for longer distances.
    line_continue_thresh: float = 0.01

    # Lines must be longer than this length in pixels, else they will not be drawn.
    # Smaller numbers can mean lots of short lines, but will represent small details more faithfully.
    # Larger numbers can result in an aesthetically-pleasing "sketchy" look - losing small details.
    min_line_length: int = 21

    # Sets the amount of angle change before a line will be terminated.
    # Small numbers mean many short lines; large numbers mean longer lines, but they cut corners.
    # For curves, this is how far a stroke may bend away from its starting direction.
    # None means "use the default for the chosen style".
    max_curve_angle_deg: float | None = None

    # When drawing / extending lines, each new pixel contributes to the line direction, via a low-pass filter with this
    # attack value.  Higher numbers mean lines can turn faster.
    lpf_atk: float = 0.05

    # Curves only: distance in pixels between the points a stroke's Bezier spline passes through.
    # Smaller values hug the traced edge more closely; larger values give smoother, looser strokes.
    curve_point_spacing: float = 12.0

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the monochrome image as a preprocessing step.
    # This can help bring out more details, especially if the input image has large areas of bright and dark.
    use_clahe: bool = True
    clahe_kernel_size: int = 32

    # Apply a Gaussian blur as a preprocessing step.
    # This can help generate longer lines if the input image is noisy or otherwise high-frequency.
    use_gaussian_blur: bool = True
    gaussian_kernel_size: float = 1

    def curve_angle_rad(self):
        if self.max_curve_angle_deg is not None:
            return math.radians(self.max_curve_angle_deg)
        return math.radians(DEFAULT_MAX_CURVE_ANGLE_DEG[self.style])


DEFAULT_MAX_CURVE_ANGLE_DEG = {"lines": 20.0, "curves": 45.0}

# ---------------------------------------------------------------------------------------------------------------------
# Utility functions

def rgb2gray(img):
    if img.ndim == 2:
        return img.astype(float)
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def bilinearInterpolate(img, x, y):
    xfloat = x - math.floor(x)
    yfloat = y - math.floor(y)

    xfloor = math.floor(x)
    yfloor = math.floor(y)

    xceil = math.ceil(x)
    yceil = math.ceil(y)

    if xfloor < 0:
        xfloor = 0
    if xceil >= img.shape[1]:
        xceil = img.shape[1] - 1

    if yfloor < 0:
        yfloor = 0
    if yceil >= img.shape[0]:
        yceil = img.shape[0] - 1

    topLeft = img[int(yfloor), int(xfloor)]
    topRight = img[int(yfloor), int(xceil)]
    bottomLeft = img[int(yceil), int(xfloor)]
    bottomRight = img[int(yceil), int(xceil)]

    topMid = xfloat * topRight + (1 - xfloat) * topLeft
    botMid = xfloat * bottomRight + (1 - xfloat) * bottomLeft

    mid = yfloat * botMid + (1 - yfloat) * topMid

    return mid

def getLineFromGradient(img, px, py, gradx, grady, params):
    angle = math.atan2(grady[py, px], gradx[py, px])
    maxAngle = params.curve_angle_rad()

    # Attempt to grow the line as much as possible.
    # The line can continue forever at the given angle, until:
    #   - the Sobel value changes too much
    #   - the gradient direction changes too much (more than MAX_CURVE_ANGLE_DEG)

    len_left = 0
    len_right = 0

    startx = px
    starty = py
    endx = px
    endy = py

    mangle = angle

    # grow the "start" side (calling it "left" because reasons)
    while 0 < starty < img.shape[0] - 1 and 0 < startx < img.shape[1] - 1 \
            and bilinearInterpolate(img, startx, starty) > params.line_continue_thresh * img[py, px]:

        len_left += 1

        # Recalculate angle.  This allows lines to "follow" curves, at least a little.
        cangle = math.atan2(grady[int(round(starty)), int(round(startx))],
                            gradx[int(round(starty)), int(round(startx))])

        # low pass filtered angle update
        mangle = mangle * (1 - params.lpf_atk) + cangle * params.lpf_atk

        if abs(angle - mangle) > maxAngle:
            break

        startx = px + len_left * math.sin(mangle)
        starty = py - len_left * math.cos(mangle)

    mangle = angle

    # grow the "end" side (calling it "right" because reasons)
    while 0 < endy < img.shape[0] - 1 and 0 < endx < img.shape[1] - 1 \
            and bilinearInterpolate(img, endx, endy) > params.line_continue_thresh * img[py, px]:

        len_right += 1

        # Recalculate angle.  This allows lines to "follow" curves, at least a little.
        cangle = math.atan2(grady[int(round(endy)), int(round(endx))],
                            gradx[int(round(endy)), int(round(endx))])

        # low pass filtered angle update
        mangle = mangle * (1 - params.lpf_atk) + cangle * params.lpf_atk

        if abs(angle - mangle) > maxAngle:
            break

        endx = px - len_right * math.sin(mangle)
        endy = py + len_right * math.cos(mangle)

    return int(round(startx)), int(round(starty)), int(round(endx)), int(round(endy)), (len_left + len_right + 1)

def edgeTangent(gradx, grady, x, y):
    # Unit vector along the edge (perpendicular to the image gradient), or None where the gradient vanishes.
    # Matches the direction convention of getLineFromGradient: (sin(a), -cos(a)) for a = atan2(gy, gx).
    gx = gradx[int(round(y)), int(round(x))]
    gy = grady[int(round(y)), int(round(x))]
    norm = math.hypot(gx, gy)
    if norm < 1e-12:
        return None
    return np.array([gy / norm, -gx / norm])

def traceCurveFromGradient(img, px, py, gradx, grady, params):
    # Walk along the edge from the seed pixel in both directions, one pixel per step, steering with the local edge
    # tangent.  Unlike getLineFromGradient, the path itself bends, so the result is a polyline that follows curves.
    # Stops under the same conditions as lines: the edge gets too weak, or the direction drifts too far from where
    # it started.
    maxAngle = params.curve_angle_rad()
    cosMaxAngle = math.cos(maxAngle)
    thresh = params.line_continue_thresh * img[py, px]

    seed = np.array([float(px), float(py)])
    t0 = edgeTangent(gradx, grady, px, py)
    if t0 is None:
        return seed[np.newaxis, :]

    halves = []
    for d0 in (t0, -t0):
        d = d0.copy()
        pos = seed.copy()
        pts = []
        while True:
            pos = pos + d
            x, y = pos
            if not (0 < y < img.shape[0] - 1 and 0 < x < img.shape[1] - 1):
                break
            if bilinearInterpolate(img, x, y) <= thresh:
                break
            pts.append(pos)

            t = edgeTangent(gradx, grady, x, y)
            if t is not None:
                # The tangent is only defined up to sign; keep heading the way we're already going.
                if np.dot(t, d) < 0:
                    t = -t
                # low pass filtered direction update
                d = d * (1 - params.lpf_atk) + t * params.lpf_atk
                d = d / np.linalg.norm(d)

            if np.dot(d, d0) < cosMaxAngle:
                break
        halves.append(pts)

    return np.array(halves[1][::-1] + [seed] + halves[0])

def resamplePolyline(points, spacing):
    # Evenly spaced points along a polyline, always keeping both endpoints.
    seglen = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seglen)])
    total = arclen[-1]
    n = max(2, int(round(total / spacing)) + 1)
    s = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(s, arclen, points[:, 0]), np.interp(s, arclen, points[:, 1])])

def catmullRomToBezier(points):
    # Convert a uniform Catmull-Rom spline through `points` into cubic Bezier segments, shape (N-1, 4, 2).
    # The spline passes through every point and is smooth (C1) at each of them.
    padded = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    p0 = padded[1:-2]
    p3 = padded[2:-1]
    c1 = p0 + (padded[2:-1] - padded[:-3]) / 6.0
    c2 = p3 - (padded[3:] - padded[1:-2]) / 6.0
    return np.stack([p0, c1, c2, p3], axis=1)

def sampleBeziers(segments, step=0.5):
    # Densely sample cubic Bezier segments, roughly `step` pixels apart.
    samples = []
    for p0, c1, c2, p3 in segments:
        # control polygon length is an upper bound on the arc length
        polyLen = np.linalg.norm(c1 - p0) + np.linalg.norm(c2 - c1) + np.linalg.norm(p3 - c2)
        t = np.linspace(0.0, 1.0, max(2, int(math.ceil(polyLen / step)) + 1))[:, np.newaxis]
        mt = 1 - t
        samples.append(mt ** 3 * p0 + 3 * mt ** 2 * t * c1 + 3 * mt * t ** 2 * c2 + t ** 3 * p3)
    return np.vstack(samples)

def bezierPathData(segments):
    d = ["M %.2f,%.2f" % tuple(segments[0][0])]
    for _, c1, c2, p3 in segments:
        d.append("C %.2f,%.2f %.2f,%.2f %.2f,%.2f" % (tuple(c1) + tuple(c2) + tuple(p3)))
    return " ".join(d)

# ---------------------------------------------------------------------------------------------------------------------
# Main pipeline

def preprocess(baseImage, params):
    baseImageGray = rgb2gray(baseImage)

    # normalize 0..1
    normImgGray = baseImageGray - baseImageGray.min()
    if normImgGray.max() == 0:
        raise ValueError("input image is a single flat color; there are no edges to draw")
    normImgGray = normImgGray / normImgGray.max()

    # CLAHE (brings out details)
    if params.use_clahe:
        normImgGray = skimage.exposure.equalize_adapthist(normImgGray, kernel_size=params.clahe_kernel_size)

    # Gaussian blur (gets rid of details)
    if params.use_gaussian_blur:
        normImgGray = scipy.ndimage.gaussian_filter(normImgGray, params.gaussian_kernel_size)

    return normImgGray

def draw(normImgGray, params, dwg, show=False):
    if show:
        import matplotlib.pyplot as plt

    sobelDx = scipy.ndimage.sobel(normImgGray, 0)  # horizontal
    sobelDy = scipy.ndimage.sobel(normImgGray, 1)  # vertical
    mag = np.hypot(sobelDx, sobelDy)

    # where the image is locally darker in low frequencies, increase the probability of drawing a line.
    imgBlur = scipy.ndimage.gaussian_filter(normImgGray, 2)
    mag = np.multiply(mag, imgBlur.max() - imgBlur)

    # turn mag into a proper probability distribution function
    mag = mag / np.sum(mag)

    magGradY, magGradX = np.gradient(normImgGray)

    if show:
        plt.imshow(normImgGray, cmap='gray')
        plt.title("preprocessed image")
        plt.show()

        plt.imshow(mag)
        plt.colorbar()
        plt.title("edge magnitude")
        plt.show()

        plt.imshow(magGradX)
        plt.title("gradient x")
        plt.show()

        plt.imshow(magGradY)
        plt.title("gradient y")
        plt.show()

    # Sample a lot of edges

    outImg = np.zeros(mag.shape, dtype=np.uint8)

    initmaxp = mag.max()
    cmax = initmaxp
    i = 0
    numStrokes = 0

    llacc = 0.0
    llcnt = 0
    minll = 1e9
    maxll = 0

    while cmax > initmaxp * params.termination_ratio:
        i = i + 1
        if i % 250 == 0:
            print("Max P: ", mag.max(), " term at:", initmaxp * params.termination_ratio)
            if llcnt > 0:
                print("Line Stats: N=", llcnt, "length: min", minll, "mean", llacc / llcnt, "max", maxll)
            llacc = 0
            llcnt = 0
            minll = 99999
            maxll = 0

        pIdxRow, pIdxCol = np.unravel_index(mag.argmax(), mag.shape)
        pIdxRow = int(pIdxRow)
        pIdxCol = int(pIdxCol)

        cmax = mag[pIdxRow, pIdxCol]

        if params.style == "lines":
            (lstartx, lstarty, lendx, lendy, totalLength) = getLineFromGradient(mag, pIdxCol, pIdxRow,
                                                                                magGradX, magGradY, params)
        else:
            path = traceCurveFromGradient(mag, pIdxCol, pIdxRow, magGradX, magGradY, params)
            totalLength = len(path)

        if totalLength < params.min_line_length:
            # this is too short, and is not worth drawing
            # We don't want to shorten other lines, so replace this peak with the (guaranteed smaller) mean of its neighbors
            acc = 0.0
            cnt = 0
            if pIdxRow + 1 < mag.shape[0]:
                acc = acc + mag[pIdxRow + 1, pIdxCol]
                cnt = cnt + 1

            if pIdxCol + 1 < mag.shape[1]:
                acc = acc + mag[pIdxRow, pIdxCol + 1]
                cnt = cnt + 1

            if pIdxRow - 1 >= 0:
                acc = acc + mag[pIdxRow - 1, pIdxCol]
                cnt = cnt + 1

            if pIdxCol - 1 >= 0:
                acc = acc + mag[pIdxRow, pIdxCol - 1]
                cnt = cnt + 1

            mag[pIdxRow, pIdxCol] = acc / cnt

            continue

        # collect line statistics:

        # accumulator for mean & line count
        llacc = llacc + totalLength
        llcnt = llcnt + 1
        numStrokes = numStrokes + 1

        # min/max line lengths
        if totalLength < minll:
            minll = totalLength

        if totalLength > maxll:
            maxll = totalLength

        if params.style == "lines":
            # draw this line in the SVG image, too
            dwg.add(dwg.line((lstartx, lstarty), (lendx, lendy), stroke=svgwrite.rgb(0, 0, 0, '%')))

            # draw the line on the edge image and preview images.

            rr, cc, val = skimage.draw.line_aa(lstarty, lstartx, lendy, lendx)
            rrd, ccd = skimage.draw.line(lstarty, lstartx, lendy, lendx)

            rr = np.clip(rr, 0, mag.shape[0] - 1)
            cc = np.clip(cc, 0, mag.shape[1] - 1)
            rrd = np.clip(rrd, 0, mag.shape[0] - 1)
            ccd = np.clip(ccd, 0, mag.shape[1] - 1)

            # Draw the line in the preview image, for visualization purposes
            outImg[rrd, ccd] = 255

            # Draw the line in the edge magnitude image
            mag[rr, cc] = 0
        else:
            segments = catmullRomToBezier(resamplePolyline(path, params.curve_point_spacing))

            # draw this curve in the SVG image, too
            dwg.add(dwg.path(d=bezierPathData(segments), stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none'))

            # draw the curve on the edge image and preview images.
            samples = sampleBeziers(segments)
            xs = samples[:, 0]
            ys = samples[:, 1]

            # Draw the curve in the preview image, for visualization purposes
            rrd = np.clip(np.round(ys).astype(int), 0, mag.shape[0] - 1)
            ccd = np.clip(np.round(xs).astype(int), 0, mag.shape[1] - 1)
            outImg[rrd, ccd] = 255

            # Knock out the curve in the edge magnitude image, ~2px wide like the anti-aliased lines above
            for rr in (np.floor(ys), np.ceil(ys)):
                for cc in (np.floor(xs), np.ceil(xs)):
                    mag[np.clip(rr.astype(int), 0, mag.shape[0] - 1),
                        np.clip(cc.astype(int), 0, mag.shape[1] - 1)] = 0

        mag[pIdxRow, pIdxCol] = 0  # also knock down the peak intensity that created this line, because the lines can move

    outImg = 255 - outImg

    if show:
        plt.imshow(outImg, cmap='gray')
        plt.title("result")
        plt.show()

    return outImg, numStrokes

def parseArgs(argv=None):
    d = Params()
    parser = argparse.ArgumentParser(description="Turn images into drawings for a pen plotter.")
    parser.add_argument("input", help="input image (any format scikit-image can read)")
    parser.add_argument("-o", "--output", help="output SVG path (default: input path with .svg extension)")
    parser.add_argument("--preview", metavar="PNG", help="also save a raster preview of the drawing to this path")
    parser.add_argument("--show", action="store_true", help="show intermediate images and the result in matplotlib")

    g = parser.add_argument_group("style")
    g.add_argument("--style", choices=["curves", "lines"], default=d.style,
                   help="curves: smooth Bezier strokes that follow edges; lines: straight strokes (default: %(default)s)")
    g.add_argument("--termination-ratio", type=float, default=d.termination_ratio,
                   help="stop when the strongest remaining edge falls below this fraction of the initial peak; "
                        "smaller draws more strokes (default: %(default).3f)")
    g.add_argument("--line-continue-thresh", type=float, default=d.line_continue_thresh,
                   help="extend a stroke until edge intensity drops below this fraction of its seed; "
                        "smaller makes longer strokes (default: %(default)s)")
    g.add_argument("--min-line-length", type=int, default=d.min_line_length,
                   help="discard strokes shorter than this many pixels (default: %(default)s)")
    g.add_argument("--max-curve-angle", type=float, default=None, dest="max_curve_angle_deg",
                   help="degrees a stroke may turn before it ends (default: %s for lines, %s for curves)"
                        % (DEFAULT_MAX_CURVE_ANGLE_DEG["lines"], DEFAULT_MAX_CURVE_ANGLE_DEG["curves"]))
    g.add_argument("--lpf-atk", type=float, default=d.lpf_atk,
                   help="how quickly a stroke's direction follows the edge; higher turns faster (default: %(default)s)")
    g.add_argument("--curve-point-spacing", type=float, default=d.curve_point_spacing,
                   help="curves only: pixels between spline points; larger is smoother and looser "
                        "(default: %(default)s)")

    g = parser.add_argument_group("preprocessing")
    g.add_argument("--no-clahe", action="store_false", dest="use_clahe", help="disable adaptive histogram equalization")
    g.add_argument("--clahe-kernel-size", type=int, default=d.clahe_kernel_size,
                   help="CLAHE kernel size in pixels (default: %(default)s)")
    g.add_argument("--no-blur", action="store_false", dest="use_gaussian_blur", help="disable Gaussian blur")
    g.add_argument("--blur-sigma", type=float, default=d.gaussian_kernel_size, dest="gaussian_kernel_size",
                   help="Gaussian blur sigma in pixels (default: %(default)s)")

    args = parser.parse_args(argv)
    params = Params(**{k: v for k, v in vars(args).items() if k in Params.__dataclass_fields__})
    return args, params

def main(argv=None):
    args, params = parseArgs(argv)
    outputPath = args.output or os.path.splitext(args.input)[0] + ".svg"

    baseImage = skimage.io.imread(args.input)
    normImgGray = preprocess(baseImage, params)
    height, width = normImgGray.shape

    dwg = svgwrite.Drawing(outputPath, size=(width, height), profile='tiny')
    dwg.viewbox(0, 0, width, height)

    outImg, numStrokes = draw(normImgGray, params, dwg, show=args.show)

    dwg.save()
    print("Wrote %d %s to %s" % (numStrokes, params.style, outputPath))

    if args.preview:
        skimage.io.imsave(args.preview, outImg, check_contrast=False)
        print("Wrote preview to", args.preview)


if __name__ == "__main__":
    main()
