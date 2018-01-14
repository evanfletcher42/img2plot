# img2plot
Turn images into drawings for a pen plotter!

## What is this?
img2plot is a script that attempts to make artistic line drawings out of input images, for drawing on a pen plotter.  The script can read most image formats and saves outputs as an SVG file.  

Unlike other image-to-pen-plotter programs, img2plot tries to mimic human drawing styles with lines - instead of drawing densities with stippling, circles, or waves, the output will be lines following edges and sketchy representations of gradients.  The intention is to mimic the appearance of a quick notebook pen sketch.  

This was originally written with a Silhouette Cameo in mind, but many other plotters, laser cutters, or other devices that take vector input ought to work fine.

## How do I use it?
This was written for Python 2.7 and requires the `matplotlib`, `numpy`, `scikit-image`, and `svgwrite` packages.  All of these can be installed via pip, but if you're on Windows you may need to install scikit-image via a binary Python wheel - see instructions [here](http://scikit-image.org/docs/dev/install.html).  

Once dependencies are set up:
1. Clone or download this repository
2. Edit the input/output paths in the first few lines of the script
3. Tune any configuration parameters as needed
4. Run, and re-tune as needed
5. Load the SVG into a plotter program of your choice!  (Or anything else, really.)

## Results

![betta](https://raw.githubusercontent.com/evanfletcher42/img2plot/master/readme-imgs/betta.jpg)
![revali](https://raw.githubusercontent.com/evanfletcher42/img2plot/master/readme-imgs/revali.jpg)
![dunwall](https://raw.githubusercontent.com/evanfletcher42/img2plot/master/readme-imgs/dunwall.jpg)
![printed](https://raw.githubusercontent.com/evanfletcher42/img2plot/master/readme-imgs/img2plot.jpg)

## Future Plans
* Draw smooth Bezier curves instead of lines - would change the art style a little, but would look more "human"
* Port to C++ / OpenCV for speed on large images
