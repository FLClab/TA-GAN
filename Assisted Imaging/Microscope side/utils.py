import sys

from matplotlib import pyplot

import numpy

from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

from skimage import draw, filters


def avg_area(img, radius, point):
    '''Compute the average of the area around the given point.
    
    :param img: An image (2d array).
    :param radius: Vertical and horizontal radius to consider around point.
    :param point: Coordinates tuple.
    
    :returns: The average of the area around the given point.
    '''
    x, y = int(point[0]), int(point[1])
    x_max, y_max = img.shape[1] - 1, img.shape[0] - 1
    assert y >= 0 and x >= 0 and y <= y_max and x <= x_max
    
    y_start, x_start = max(0, y-radius), max(0, x-radius)
    y_end, x_end = min(y_max, y_start+2*radius), min(x_max, x_start+2*radius)
    return numpy.mean(img[y_start:y_end+1, x_start:x_end+1])


def avg_signal(signal, radius, points):
    '''Compute the estimated signal using the average over areas around all the
    given points.
    
    :param signal: An image (2d array).
    :param radius: Vertical and horizontal radius to consider around points.
    :param points: List of coordinates tuples.
    
    :returns: The estimated signal.
    '''
    avg_areas = [avg_area(signal, radius, point) for point in points]
    return numpy.mean(avg_areas)


def gaussian_fit(img, start, end):
    values = []
    for delta in [-1, 0, 1]:
        cc, rr = draw.line(*start, *end)
        rr += delta
        positions = numpy.sqrt((rr - start[0])**2 + (cc - start[1])**2)
        values.append(img[rr, cc])
    values = numpy.asarray(values)
    avg_values = numpy.mean(values, axis=0)
    y0 = numpy.mean(values)
    mu = numpy.mean(positions)
    sigma = numpy.std(positions)
    a = numpy.max(avg_values) * 2 * sigma
    try:
        popt = curve_fit(gauss2, positions, avg_values, p0=[y0, a, mu, sigma])[0]
    except (RuntimeError, TypeError, NameError) as err:
        print("Gaussian fit failed")
        print(err)
        pyplot.figure("Failed to fit these data")
        pyplot.plot(positions, avg_values, "bo")
        pyplot.show(block=True)
        popt = None
    return popt, positions, avg_values


def points2regions(points, pixelsize, resolution):
    '''Return the regions corresponding to the selected points.
    '''
    x_pixels, y_pixels = resolution
    x_size, y_size = pixelsize
    print("points2regions")
    print("resolution", x_pixels, "(x),", y_pixels, "(y)")
    print("pixelsize", x_size, "(x),", y_size, "(y)")
    return [((x-x_pixels/2)*x_size, (y-y_pixels/2)*y_size) for (x, y) in points]


def rect2regions(rectangles, pixelsize):
    ''' Outputs the new window size from the selected rectangles
    '''
    x_size, y_size = pixelsize
    new_window_size = []

    for (start_point, end_point) in rectangles:
        if end_point[0] > start_point[0] and end_point[1] > start_point[1]:
            new_window_size.append(((end_point[0]-start_point[0])*x_size, (end_point[1]-start_point[1])*y_size))
        elif end_point[0] < start_point[0] and end_point[1] > start_point[1]:
            new_window_size.append(((start_point[0]-end_point[0])*x_size, (end_point[1]-start_point[1])*y_size))
        elif end_point[0] < start_point[0] and end_point[1] < start_point[1]:
            new_window_size.append(((start_point[0]-end_point[0])*x_size, (start_point[1]-end_point[1])*y_size))
        else:
            new_window_size.append(((end_point[0]-start_point[0])*x_size, (start_point[1]-end_point[1])*y_size))
    return new_window_size
    # return [((end_point[0]-start_point[0])*x_size, (end_point[1]-start_point[1])*y_size) for (start_point, end_point) in rectangles]


def get_rect_center(rectangles, pixelsize, resolution):
    '''Outputs the center of the selected rectangles
    '''
    x_pixels, y_pixels = resolution
    x_size, y_size = pixelsize
    rect_center = []
    print(rectangles)
    for start_point, end_point in rectangles:
        print(start_point)
        if end_point[0] > start_point[0] and end_point[1] > start_point[1]:
            rect_center.append((((start_point[0]+(end_point[0]-start_point[0])/2)-x_pixels/2)*x_size, ((start_point[1]+(end_point[1]-start_point[1])/2)-y_pixels/2)*y_size))
        elif end_point[0] < start_point[0] and end_point[1] > start_point[1]:
            rect_center.append((((end_point[0]+(start_point[0]-end_point[0])/2)-x_pixels/2)*x_size, ((start_point[1]+(end_point[1]-start_point[1])/2)-y_pixels/2)*y_size))
        elif end_point[0] < start_point[0] and end_point[1] < start_point[1]:
            rect_center.append((((end_point[0]+(start_point[0]-end_point[0])/2)-x_pixels/2)*x_size, ((end_point[1]+(start_point[1]-end_point[1])/2)-y_pixels/2)*y_size))
        else:
            rect_center.append((((start_point[0]+(end_point[0]-start_point[0])/2)-x_pixels/2)*x_size, ((end_point[1]+(start_point[1]-end_point[1])/2)-y_pixels/2)*y_size))
    # return [(((start_point[0]+(end_point[0]-start_point[0])/2)-x_pixels/2)*x_size, ((start_point[1]+(end_point[1]-start_point[1])/2)-y_pixels/2)*y_size) for (start_point, end_point) in rectangles]
    return rect_center

def gauss(x, a, mu, sigma):
    return a * numpy.exp(-(x-mu)**2/(2*sigma**2))


def gauss2(x, y0, a, mu, sigma):
    w = 2 * sigma
    return y0 + (a / w * numpy.sqrt(numpy.pi/2)) * numpy.exp(-2*(x-mu)**2/w**2)
    

def get_closer(last_options, last_preferred, options, metric):
    dists = cdist(options, [last_options[last_preferred]], metric=metric)
    return numpy.argmin(dists)


def get_foreground(img):
    val = filters.threshold_otsu(img)
    return img > val


if __name__ == "__main__":
    import skimage.io
    path = "Test foreground background/35"
    img = skimage.io.imread(path+".tiff")
    fg = get_foreground(img)
    img_fg = numpy.zeros_like(img)
    img_fg[fg] = 255
    skimage.io.imsave(path+"_fg.pdf", img_fg)