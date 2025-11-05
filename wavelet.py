import numpy
import math
import sys
import xlsxwriter
import os
from typing import List
from skimage.measure import regionprops, label
from skimage import feature
from matplotlib import pyplot
from math import atan2
from skimage import morphology
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist


"""
This script is intended to perform a cluster analysis based on the paper

Mapping molecular assemblies with fluorescence microscopy and object-based spatial statistics
Thibault Lagache, Alexandre Grassart, Stéphane Dallongeville, Orestis Faklaris, 
Nathalie Sauvonnet, Alexandre Dufour, Lydia Danglot & Jean-Christophe Olivo-Marin

The first step is to perform the detection of the clusters with wevelet transformation 
of the image and statistical thresholding of wavelets coefficients.
The second sted is to characterize the spatial distribution of the clusters using
a Marked Point Process.
The third and final step is to characterize the spatial relations between the clusters.
The Ripley's K Function will be of great help.
"""

def filter_spots(intensity_image, mask):
        """
        Removes the spots that are too small or too linear (using parameters min_size and min_axis)
        :param mask: 3D binary mask of spots to filter
        :return: Filtered 3D binary mask
        """
        out_mask = numpy.copy(mask).astype(bool)
        img = morphology.remove_small_objects(out_mask, min_size=30)
        mask_lab, num = label(img, connectivity=1, return_num=True)
        mask_props = regionprops(mask_lab, intensity_image=intensity_image)
        for p in mask_props:
            if p.mean_intensity < 3:
                mask_lab[mask_lab == p.label] = 0
            if p.minor_axis_length < 3:
                mask_lab[mask_lab == p.label] = 0

        out_mask = mask_lab > 0
        return out_mask

def detect_spots(img: numpy.ndarray, J_list: List[int] = (3,4), scale_threshold: float = 200) -> numpy.ndarray:
    """
    Detects spots in an image using the wavelet transform and statistical thresholding of wavelet coefficients.
    """
    detector = DetectionWavelets(img, J_list, scale_threshold)
    spots_image = detector.computeDetection()
    filtered_spots = filter_spots(img, spots_image)
    return filtered_spots


class DetectionWavelets:
    """
    This is based on the paper
        "Extraction of spots in biological images using multiscale products"
    All functions from the Java code for the Icy Spot Detector plugin are implemented here.
    """

    def __init__(self, img, J_list=(3,4), scale_threshold=200):
        """Init function
        :param img: A numpy 2D array
        :param J_list: List of all chosen scales
        :param scale_threshold: Percent modifier of wavelet image threshold
        """
        self.img = img
        self.J = max(J_list)
        self.J_list = J_list
        self.scale_threshold = scale_threshold

    def computeDetection(self):
        """
        Computes the binary correlation image
        :return image_out: numpy array representing the binary image
        """

        data_in = numpy.copy(self.img).astype('float32')
        scales = self.b3WaveletScales2D(data_in)
        coefficients = self.b3WaveletCoefficients2D(scales, data_in)
        for i in range(len(coefficients)-1):
            coefficients[i] = self.filter_wat(coefficients[i], i)
        coefficients[-1] *= 0

        binary_detection_result = self.spot_construction(coefficients)
        binary_detection_result[binary_detection_result != 0] = 255

        return binary_detection_result.astype('uint8')

    def b3WaveletScales2D(self, data_in):
        """
        Computes the convolution images for scales J
        :param data_in: Base image as 1D list
        :return res_array: List of convoluted images as 1D lists
        """

        prev_array = data_in.copy()
        res_array = []

        for s in range(1, self.J+1):
            stepS = 2**(s-1)

            current_array = self.filter_and_swap(prev_array, stepS)

            if s == 1:
                prev_array = current_array
            else:
                tmp = current_array
                prev_array = tmp

            current_array = self.filter_and_swap(prev_array, stepS)
            tmp = current_array
            prev_array = tmp

            res_array.append(prev_array)

        return res_array


    def b3WaveletCoefficients2D(self, scale_coefficients, original_image):
        """
        Computes the difference between consecutive wavelet transform images
        :param scale_coefficients: List of  convoluted images as 2D numpy arrays
        :param original_image: Original image as 1D list
        :return wavelet_coefficients: List of coefficient images 2D numpy arrays
        """

        wavelet_coefficients = []
        iter_prev = original_image.copy()
        for j in range(self.J):
            iter_current = scale_coefficients[j]
            w_coefficients = iter_prev - iter_current
            wavelet_coefficients.append(w_coefficients)
            iter_prev = iter_current
        wavelet_coefficients.append(scale_coefficients[self.J-1])
        wavelet_coefficients = numpy.stack(wavelet_coefficients, 0)
        return wavelet_coefficients

    def filter_wat(self, data, depth):
        """
        Applies a threshold on the coefficient images
        :param data: image data
        :param depth: number of scale
        :param width: image width
        :param height: image height
        :return output: filtered image
        """

        output = data.copy()
        lambdac = []

        for i in range(self.J + 2):
            lambdac.append(numpy.sqrt(2 * numpy.log(data.size / (1 << (2 * i)))))

        # mad
        size = data.size
        mean = numpy.mean(data)
        s = data - mean
        a = numpy.sum(numpy.abs(s))

        mad = a / size

        dcoeff = self.scale_threshold

        coeff_thr = (lambdac[depth + 1] * mad) / dcoeff

        output[output < coeff_thr] = 0

        return output

    def spot_construction(self, input_coefficients):
        """
        Reconstructs correlation image with multiscale product
        :param input_coefficients: 3D numpy of array wavelet coefficient images
        :return output: Correlation image as 2D numpy array
        """
        J_array = numpy.array(self.J_list)-1
        #zero_coords = numpy.prod(input_coefficients[J_array], axis=0) > 0

        output = numpy.prod(input_coefficients[J_array], axis=0)
        #output *= zero_coords
        return output

    @staticmethod
    def filter_and_swap(array_in, stepS):
        kernel = numpy.array([1/16, 1/4, 3/8, 1/4, 1/16])
        inter = numpy.array([1, 2, 3, 4])
        for s in range(stepS):
            if s > 0:
                kernel = numpy.insert(kernel, inter, 0)
                inter = inter + numpy.array([1, 2, 3, 4])
        kernel_base = numpy.zeros((kernel.size, kernel.size))
        kernel_base[int((kernel.size-1)/2)] = kernel
        new_img = convolve(array_in, kernel_base)
        new_img = numpy.transpose(new_img)

        return new_img


class SpatialDistribution:
    """ This is based on the Marked Point Process as shown in the cited above paper
    Marked is the attributes of the cluster (shape, size, color)
    Point Process is the position of the clusters (centroid)

    :returns : A list of tuple (regionprops, area, centroid)
    NOTE. Centroid is (y, x) coordinates
    """
    def __init__(self, prob_map, img, cs=0, min_axis=1):
        """This is the init function
        :param prob_map: A 2D numpy array of boolean detected clusters
        :param img: numpy array, image data used for intensity weighting for centroid
        :param cs: Minimum area of a cluster
        """
        self.img = img
        self.P = prob_map
        self.P[self.P > 0] = 1
        self.cs = cs
        self.min_axis = min_axis

    def mark(self):
        """
        This function creates the Marked Point Processed of every cluster
        :return mark: List of every spot in the current channel as (regionprops, area, centroid) tuples
        """

        # Label each spot for regionprops
        labels = label(self.P, connectivity=2)

        props = regionprops(labels, intensity_image=self.img)
        mark = []
        for p in props:
            labimage = p.image
            min_distance = int(0.08 / 0.015) // 2 + 1 
            peaks = feature.peak_local_max(self.img, min_distance=min_distance, exclude_border=False, labels=labimage)

            s = p.area
            if s > 0:
                try:
                    try:
                        cent_int = p.weighted_centroid
                        int(cent_int[0])
                    except ValueError:
                        cent_int = p.centroid

                    # Reject small and linear clusters
                    if s >= self.cs and p.minor_axis_length >= self.min_axis \
                            and p.major_axis_length >= self.min_axis\
                            and p.perimeter > 0:
                        mark.append(
                            (p, s, cent_int, len(peaks))  # p is the regionprops for a spot: every characteristic can be accessed
                                              # as an attribute e.g. p.eccentricity
                        )
                except IndexError:
                    print('Index error: spot was ignored.')

        return mark

    def poly_area(self, x, y):
        """ This function computes the area of a cluster
        :param x: A numpy array of x coordinates
        :param y: A numpy array of y coordinates
        """
        return 0.5*numpy.abs(numpy.dot(x,numpy.roll(y,1))-numpy.dot(y,numpy.roll(x,1)))


class SpatialRelations:
    """ This is based on the the paper cited above
    To characterise the spatial relations between two populations A1 (green) and A2 (red) of
    objects (spots or localisations), we use the Ripley’s K function, a gold standard for analysing
    the second-order properties (i.e., distance to neighbours) of point processes.
    """
    def __init__(self, MPP1, MPP2, sROI, roivolume, poly, img, n_rings, step, filename):
        """ The init function
        :param MPP1: The marked point process of every clusters in the first channel
        :param MPP2: The marked point process of every clusters in the second channel
        :param sROI: A tuple (y, x) of the window size
        :param roivolume: Volume of the detected ROI
        :param poly: List of vertices of the ROI
        :param img: numpy array containing image data
        :param n_rings: int, number of rings around each spot
        :param step: numeric, width of each ring
        :param filename: string, name of the current image, used for naming output files
        """

        self.img = img
        self.filename = filename

        self.MPP1 = MPP1
        self.MPP2 = MPP2

        self.ROIarea = roivolume
        self.poly = poly

        # self.MPP1_ROI = self.mpp_in_contours(self.MPP1)
        # self.MPP2_ROI = self.mpp_in_contours(self.MPP2)
        self.MPP1_ROI = self.MPP1
        self.MPP2_ROI = self.MPP2

        self.sROI = sROI
        self.max_dist = n_rings*step
        self.pas = step

        self.rings = numpy.array([r*step for r in range(0, n_rings+1, 1)])
        self.imgw1 = self.image_windows(self.MPP1_ROI)
        self.imgw2 = self.image_windows(self.MPP2_ROI)

        self.neighbors = self.nearest_neighbors()

        self.distance_fit, self.N_fit = self.dist_fit()

    @staticmethod
    def matrix_dist(X, Y):
        """
        Calculates the distance between every point contained in matrices
        :param X: numpy array with shape (len(MPP1), 2), coordinates of every points in channel 1
        :param Y: numpy array with shape (len(MPP2), 2), coordinates of every points in channel 2
        :return: numpy array of distances between every combination of points in the two channels
        """
        return numpy.sqrt((X[:, 0] - Y[:, 0]) ** 2 + (X[:, 1] - Y[:, 1]) ** 2)

    def nearest_neighbors(self):
        """
        Finds the nearest neighbor in both MPPs of each spot and the distance between them
        :return: List of tuples (dist. to neighbor, neighbor index) for each combination of channels
        """

        mpp1_data = numpy.zeros((len(self.MPP1_ROI), 2))
        mpp2_data = numpy.zeros((len(self.MPP2_ROI), 2))

        for i in range(len(self.MPP1_ROI)):
            p, s, (y, x), n_peaks = self.MPP1_ROI[i]
            mpp1_data[i, 0] = x
            mpp1_data[i, 1] = y

        for i in range(len(self.MPP2_ROI)):
            p, s, (y, x), n_peaks = self.MPP2_ROI[i]
            mpp2_data[i, 0] = x
            mpp2_data[i, 1] = y

        # MPP1 with MPP1
        distances = []
        min_distance = []
        for x in mpp1_data:
            repmat = numpy.tile(x, (mpp1_data.shape[0], 1))
            distance = self.matrix_dist(repmat, mpp1_data)
            distance_pos = distance[numpy.where((distance > 0))]
            distances.append(distance)
            m = distance_pos.min()
            m_index = numpy.where((distance == m))[0][0]
            p2, s2, (y2, x2), n_peaks2 = self.MPP1_ROI[m_index]
            delta_x = x2 - x[0]
            delta_y = y2 - x[1]
            angle = atan2(delta_y, delta_x)
            min_distance.append((m, m_index, angle))
        min_dist_11 = min_distance

        # MPP1 with MPP2
        distances = []
        min_distance = []
        for x in mpp1_data:
            repmat = numpy.tile(x, (mpp2_data.shape[0], 1))
            distance = self.matrix_dist(repmat, mpp2_data)
            distance_pos = distance[numpy.where((distance > 0))]
            distances.append(distance)
            m = distance_pos.min()
            m_index = numpy.where((distance == m))[0][0]
            p2, s2, (y2, x2), n_peaks2 = self.MPP2_ROI[m_index]
            delta_x = x2 - x[0]
            delta_y = y2 - x[1]
            angle = atan2(delta_y, delta_x)
            min_distance.append((m, m_index, angle))
        min_dist_12 = min_distance

        # MPP2 with MPP1
        distances = []
        min_distance = []
        for x in mpp2_data:
            repmat = numpy.tile(x, (mpp1_data.shape[0], 1))
            distance = self.matrix_dist(repmat, mpp1_data)
            distance_pos = distance[numpy.where((distance > 0))]
            distances.append(distance)
            m = distance_pos.min()
            m_index = numpy.where((distance == m))[0][0]
            p2, s2, (y2, x2), n_peaks2 = self.MPP1_ROI[m_index]
            delta_x = x2 - x[0]
            delta_y = y2 - x[1]
            angle = atan2(delta_y, delta_x)
            min_distance.append((m, m_index, angle))
        min_dist_21 = min_distance

        # MPP2 with MPP2
        min_distance = []
        distances = []
        for x in mpp2_data:
            repmat = numpy.tile(x, (mpp2_data.shape[0], 1))
            distance = self.matrix_dist(repmat, mpp2_data)
            distance_pos = distance[numpy.where((distance > 0))]
            distances.append(distance)
            m = distance_pos.min()
            m_index = numpy.where((distance == m))[0][0]
            p2, s2, (y2, x2), n_peaks2 = self.MPP2_ROI[m_index]
            delta_x = x2 - x[0]
            delta_y = y2 - x[1]
            angle = atan2(delta_y, delta_x)
            min_distance.append((m, m_index, angle))
        min_dist_22 = min_distance

        return min_dist_11, min_dist_12, min_dist_21, min_dist_22

    def dist_fit(self):
        """
        Allows to generate the distance_fit and N_fit as in the paper
        :return distance_fit: List of distances for every ring
        :return N_fit: Number of rings
        """
        distance_fit = [0]
        temp = distance_fit[0]
        while temp + self.pas <= self.max_dist:
            temp += self.pas
            distance_fit.append(temp)
        N_fit = len(distance_fit)
        if N_fit == 1:
            distance_fit.append(self.max_dist)
            N_fit = len(distance_fit)

        return distance_fit, N_fit

    def image_windows(self, points):
        """
        This function implements the image windows as proposed in the java script
        version of the algorithm.
        """
        h, w = self.sROI
        imagewindows = [[[] for i in range(int(h / self.max_dist) + 1)] for j in range((int(w / self.max_dist) + 1))]
        dist_to_border = self.nearest_contour(points)
        for k in range(len(points)):
            p, s, (y, x), n_peaks = points[k]
            j, i = int(y / self.max_dist), int(x / self.max_dist)
            imagewindows[i][j].append((y, x, s, dist_to_border[k]))
        return imagewindows

    def correlation_new(self):
        """
        This function computes the G vector of Ripley's function as in Ripley2D.java
        results[][1]=K et results[][2]=moyenne distances results[][3]=moyenne
        distances^2
        :return result: numpy array in which the first row is Ripley's G matrix
        """
        result = numpy.zeros((3, self.N_fit - 1))
        delta_K = numpy.zeros(self.N_fit - 1)
        d_mean = numpy.zeros(self.N_fit - 1)
        d_mean_2 = numpy.zeros(self.N_fit - 1)
        count = numpy.zeros(self.N_fit - 1)
        ROIy, ROIx = self.sROI
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        for i in range(len(self.imgw1)):
            for j in range(len(self.imgw1[i])):
                for y1, x1, s1, db1 in self.imgw1[i][j]:
                    # d = self.distance_pl(self.poly,x1,y1)
                    # d = self.dist_to_contour(x1, y1)
                    d = db1
                    # d = min(x1, ROIx - x1, y1, ROIy - y1) # min distance from ROI
                    for k in range(max(i - 1, 0), min(i + 1, len(self.imgw1) - 1) + 1):
                        for l in range(max(j - 1, 0), min(j + 1, len(self.imgw1[i]) - 1) + 1):
                            for y2, x2, s2, db2 in self.imgw2[k][l]:
                                temp = self.distance(x1,x2,y1,y2) # distance
                                if (y1, x1, s1) != (y2, x2, s2):
                                    weight = 1 # weight
                                    if temp > d:
                                        weight = 1 - (numpy.arccos(d / temp)) / numpy.pi
                                    for m in range(1, self.N_fit):
                                        if (temp < self.distance_fit[m]) & (temp >= self.distance_fit[0]):
                                            delta_K[m - 1] += (1 / weight) * self.ROIarea / (n1 * n2)
                                            count[m - 1] += 1
                                            d_mean[m - 1] += temp
                                            d_mean_2[m - 1] += temp**2
                                            break

        for l in range(self.N_fit - 1):
            result[0, l] = delta_K[l]
            if count[l] > 0:
                result[1, l] = d_mean[l] / count[l]
                result[2, l] = d_mean_2[l] / count[l]
            else:
                result[1, l] = 0
                result[2, l] = 0
        return result

    def nearest_contour(self, mpp):
        """
        Finds the distance between every point and the ROI border
        Because the distance is point to point, the ROI must have points all along its contour for this to be accurate.
        :param mpp: Marked Point Process of a single channel (list of points)
        :return: List of distances
        """
        mpp_data = numpy.zeros((len(mpp), 2))

        poly_sum = numpy.concatenate(self.poly)

        poly_data = numpy.zeros((len(poly_sum), 2))

        for i in range(len(mpp)):
            p, s, (y, x), n_peaks = mpp[i]
            mpp_data[i, 0] = x
            mpp_data[i, 1] = y

        for i in range(len(poly_sum)):
            y, x = poly_sum[i]
            poly_data[i, 0] = x
            poly_data[i, 1] = y

        distances = []
        min_distance = []
        for x in mpp_data:
            repmat = numpy.tile(x, (poly_data.shape[0], 1))
            distance = self.matrix_dist(repmat, poly_data)
            distances.append(distance)
            m = distance.min()
            min_distance.append(m)

        return min_distance

    def dist_to_contour(self, x, y):
        '''
        (Unused alternative to nearest_contour)
        Meant to calculate the distance between a point and the edge of a ROI if
        the ROI is composed of multiple contours
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :return: Minimum distance between a point and the contour of the ROI
        '''
        dist = sys.maxsize
        min_dist = dist

        for c in self.poly:
            min_dist = self.distance_pl(c, x, y)
            min_dist = min(dist, min_dist)

        return min_dist

    def mpp_in_poly(self, MPP, poly):
        """
        This function checks which spots are in the selected polygon
        :param MPP: Marked Point Process; list of marks
        :param poly: A list of (x, y) coordinates corresponding to a polygon's (ROI) vertexes
        :return: List of marks in the ROI
        """
        MPP_ROI = []
        for point in MPP:
            y, x = point[2]

            n = len(poly)
            inside = False

            p1y, p1x = poly[0]
            for i in range(n + 1):
                p2y, p2x = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            else:
                                xints = None
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                MPP_ROI.append(point)

        return MPP_ROI

    def mpp_in_contours(self, MPP):
        """
        Calls mpp_in_poly in a loop for every region of the ROI if it's composed of multiple shapes
        :param MPP: Marked Point Process; list of marks
        :return: List of spots inside ROI
        """

        spots_in_contour = []
        for poly in self.poly:
            spots_in_contour += self.mpp_in_poly(MPP, poly)

        return spots_in_contour

    def variance_theo_delta_new(self):
        """
        This function implements the computation of the standard deviation of G
        :return result: numpy array, variance matrix
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        mu = self.mean_G()
        result = numpy.zeros(self.N_fit - 1)
        results, N_h = self.beta_correction(self.max_dist/10, 100)

        for k in range(1, self.N_fit):
            distancek_1 = self.distance_fit[k - 1]
            distancek = self.distance_fit[k]

            d2 = distancek_1**2
            d2bis = distancek**2
            e1 = numpy.pi * (distancek_1**2)
            e2 = numpy.pi * (distancek**2)

            temp_A1, temp_A2, temp_A3 = 0, 0, 0

            sum_h_a, sum_h_a_bis = 0, 0

            for i in range(len(self.imgw1)):
                for j in range(len(self.imgw1[i])):
                    for y1, x1, s1, db1 in self.imgw1[i][j]:
                        dist = db1
                        if (dist < distancek) & (dist >= self.distance_fit[0]):
                            sum_h_a += results[math.ceil(N_h * dist / distancek)]
                        else:
                            sum_h_a += 1
                        if k > 1:
                            if (dist < distancek_1) & (dist > self.distance_fit[0]):
                                sum_h_a_bis += results[math.ceil(N_h * dist / distancek_1)]
                            else:
                                sum_h_a_bis += 1

                        for m in range(max(i - 1, 0), min(i + 1, len(self.imgw1) - 1) + 1):
                            for l in range(max(j - 1, 0), min(j + 1, len(self.imgw1[i]) - 1) + 1):
                                for y2, x2, s2, db2 in self.imgw1[m][l]:
                                    distance_ij = self.distance(x1, x2, y1, y2)
                                    if distance_ij > 0:
                                        if distance_ij < 2 * distancek_1:
                                            temp1 = 2 * d2 * numpy.arccos(distance_ij / (2 * distancek_1))
                                            temp2 = 0.5 * distance_ij * numpy.sqrt(4 * d2 - distance_ij**2)
                                            temp_A1 += temp1 - temp2
                                        if distance_ij < 2 * distancek:
                                            temp1 = 2 * distancek**2 * numpy.arccos(distance_ij / (2 * distancek))
                                            temp2 = 0.5 * distance_ij * numpy.sqrt(4 * distancek**2 - distance_ij**2)
                                            temp_A2 += temp1 - temp2
                                        if distance_ij < distancek_1 + distancek:
                                            if distance_ij + distancek_1 < distancek:
                                                temp_A3 += 2 * numpy.pi * d2
                                            else:
                                                temp1 = d2 * numpy.arccos((distance_ij**2 + d2 - d2bis) / (2 * distance_ij * distancek_1))
                                                temp2 = d2bis * numpy.arccos((distance_ij**2 + d2bis - d2) / (2 * distance_ij * distancek))
                                                temp3 = 0.5 * numpy.sqrt((- distance_ij + distancek_1 + distancek)
                                                                         * (distance_ij - distancek_1 + distancek)
                                                                         * (distance_ij + distancek_1 - distancek)
                                                                         * (distance_ij + distancek_1 + distancek))
                                                temp_A3 += 2 * (temp1 + temp2 - temp3)
            I2 = (temp_A1 + temp_A2 - temp_A3 - (e1**2 / self.ROIarea + e2**2 / self.ROIarea - 2 * e1 * e2 / self.ROIarea) * (n1 * (n1 - 1))) * n2 / self.ROIarea
            I1 = (e2 * sum_h_a - e1 * sum_h_a_bis - n1 * (e2 - e1)**2 / self.ROIarea) * n2 / self.ROIarea
            result[k - 1] = (self.ROIarea / (n2 * n1))**2 * (I1 + I2)
        return result

    def intersection2D_new(self):
        """
        This function computes the A matrix
        :return A: numpy array, A matrix
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        A = numpy.zeros((self.N_fit - 1, self.N_fit - 1))
        for r in range(self.N_fit - 1):
            A[r, r] = n1
        S = numpy.zeros((self.N_fit, self.N_fit))

        for k1 in range(0, self.N_fit):
            for k2 in range(0, self.N_fit):
                m = min(self.distance_fit[k1], self.distance_fit[k2])  # create min
                M = max(self.distance_fit[k1], self.distance_fit[k2])  # create max

                for i in range(len(self.imgw1)):
                    for j in range(len(self.imgw1[i])):
                        for y1, x1, s1, db1 in self.imgw1[i][j]:
                            for o in range(max(i - 1, 0), min(i + 1, len(self.imgw1) - 1) + 1):
                                for l in range(max(j - 1, 0), min(j + 1, len(self.imgw1[i]) - 1) + 1):
                                    for y2, x2, s2, db2 in self.imgw1[o][l]:
                                        d = self.distance(x1, x2, y1, y2)
                                        if d > 0:
                                            if m + d < M:
                                                S[k1, k2] = S[k1, k2] + numpy.pi * m ** 2
                                            else:
                                                if d < (m + M):
                                                    temp1 = m ** 2 * numpy.arccos((d ** 2 + m ** 2 - M ** 2) / (2 * d * m))
                                                    temp2 = M ** 2 * numpy.arccos((d ** 2 + M ** 2 - m ** 2) / (2 * d * M))
                                                    temp3 = 0.5 * numpy.sqrt((-d + m + M) * (d + m - M) * (d - m + M) * (d + m + M))
                                                    S[k1, k2] = S[k1, k2] + temp1 + temp2 - temp3

        for i in range(self.N_fit - 1):
            for j in range(self.N_fit - 1):
                vol = numpy.pi * (self.distance_fit[j + 1] ** 2 - self.distance_fit[j] ** 2)
                A[i, j] = A[i, j] + (S[i + 1, j + 1] + S[i, j] - S[i, j + 1] - S[i + 1, j]) / vol
        return A

    def reduced_Ripley_vector(self, **kwargs):
        """
        Vector G0 from paper. This function returns the estimation of the coupling probability
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        if kwargs:
            G = kwargs["G"][0, :]
            var = kwargs["var"]
            A = kwargs["A"]
        else:
            G = self.correlation_new()[0, :]
            var = self.variance_theo_delta_new()
            A = self.intersection2D_new()
        mean = self.mean_G()

        A_b = A/n1

        G0 = numpy.dot(numpy.linalg.inv(A_b), G - mean) / numpy.sqrt(var)

        return G0

    def draw_G0(self, G0):
        """
        Displays a bar graph of G0 by ring distance with a line for the G0 threshold, like figure 1d of the paper
        :param G0: numpy array, G0 matrix
        """
        T = numpy.sqrt(2 * numpy.log(len(G0)))
        pyplot.bar(self.rings[1:], G0)
        pyplot.axhline(y=T, color='red', linestyle='dashed')
        pyplot.title("G0")
        pyplot.show()

    def main2D_corr(self, G, var, A):
        """
        Currently unused
        P calculation from the SODA code (non_parametric_object.java)
        Works; very nearly gives the same results as coupling_prob
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        delta_K = G[0,:]
        sigma_T = numpy.zeros([self.N_fit - 1])
        Num = delta_K * (n1*n2/self.ROIarea)
        mu_tab = self.mean_G() * (n1*n2/self.ROIarea)
        for p in range(self.N_fit-1):
            if p > 0:
                T_p = (numpy.sqrt(2 * numpy.log(p+1)))
            else:
                T_p = numpy.sqrt(2)
            sigma_T[p] = (n1*n2*T_p/self.ROIarea)*numpy.sqrt(var[p])

        A_matb = A * 1/n1
        A_matb_inverse = numpy.linalg.inv(A_matb)
        C = numpy.dot(A_matb_inverse, (Num - mu_tab))

        for p in range(self.N_fit-1):
            if C[p]<sigma_T[p]:
                C[p] = 0

        proba_dist = []
        for i in range(self.N_fit-1):
            if Num[i] > 0:
                proba_dist.append(C[i]/Num[i])
            else:
                proba_dist.append(0)

        return proba_dist

    def coupling_prob(self, **kwargs):
        """
        This function computes the coupling probability between the two channels
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        if kwargs:
            G = kwargs["G"][0, :]
            var = kwargs["var"]
            A = kwargs["A"]
            G0 = kwargs["G0"]
        else:
            G = self.correlation_new()[0, :]
            var = self.variance_theo_delta_new
            A = self.intersection2D_new()
            G0 = self.reduced_Ripley_vector(**kwargs)
        mean = self.mean_G()

        # T = [numpy.sqrt(2 * numpy.log(i+1)) if i > 0 else numpy.sqrt(2) for i in range(self.N_fit)]
        T = numpy.sqrt(2 * numpy.log(len(G0)))
        coupling = numpy.array([
            (numpy.sqrt(var[i]) * G0[i]) / G[i] if G0[i] > T else 0 for i in range(len(self.rings)-1)
        ])

        # Create distance matrix and coupling probability matrix using cdist
        MPP1_array = numpy.ndarray((n1, 2))
        for i, (c1, s1, (y1, x1), n_peaks1) in enumerate(self.MPP1_ROI):
            MPP1_array[i] = numpy.array([y1, x1])

        MPP2_array = numpy.ndarray((n2, 2))
        for i, (c1, s1, (y2, x2), n_peaks2) in enumerate(self.MPP2_ROI):
            MPP2_array[i] = numpy.array([y2, x2])

        dist_array = cdist(MPP1_array, MPP2_array)
        prob_array = numpy.zeros(dist_array.shape)
        for i in range(len(self.rings) - 1):
            prob_array[numpy.where(numpy.logical_and(self.rings[i] <= dist_array, dist_array < self.rings[i+1]))] = coupling[i]
        prob_array[dist_array == 0] = 0

        # Create the list of lines for the couples excel file
        n_couples = numpy.sum(prob_array > 0)
        prob_write = []
        for i in range(n1):
            for j in range(n2):
                if prob_array[i,j] > 0:
                    prob_write.append([MPP1_array[i][1],
                                       MPP1_array[i][0],
                                       MPP2_array[j][1],
                                       MPP2_array[j][0],
                                       dist_array[i,j],
                                       prob_array[i,j],
                                       i,
                                       j])

        if numpy.sum(prob_array) > 0:
            coupling_index = (numpy.sum(prob_array)/n1, numpy.sum(prob_array)/n2)
            mean_coupling_distance = numpy.sum(dist_array*prob_array)/numpy.sum(prob_array)
        else:
            coupling_index = (0,0)
            mean_coupling_distance = None

        return prob_write, {'n_spots_0': len(self.MPP1_ROI),
                            'n_spots_1': len(self.MPP2_ROI),
                            'coupling_index': coupling_index,
                            'mean_coupling_distance': mean_coupling_distance,
                            'coupling_probabilities': coupling,
                            'n_couples': n_couples}

    def data_boxplot(self, prob_write):
        """
        Plots a box plot of a spot property by channels and coupling
        :param prob_write: List of couples and their properties
        """
        fig, axs = pyplot.subplots(1,2)

        dataC = []
        dataU = []
        for p1, s1, (y1, x1), n_peaks in self.MPP1_ROI:
            coupled = False
            for xa, ya, xb, yb, dist, p, _, _ in prob_write:
                if (y1, x1) == (ya, xa) and p1.minor_axis_length > 0:
                    dataC.append(p1.eccentricity)
                    coupled = True
            if not coupled and p1.minor_axis_length > 0:
                dataU.append(p1.eccentricity)

        data1 = [dataC, dataU]

        dataC = []
        dataU = []
        for p1, s1, (y1, x1), n_peaks in self.MPP2_ROI:
            coupled = False
            for xa, ya, xb, yb, dist, p, _, _ in prob_write:
                if (y1, x1) == (yb, xb) and p1.minor_axis_length > 0:
                    dataC.append(p1.eccentricity)
                    coupled = True
            if not coupled and p1.minor_axis_length > 0:
                dataU.append(p1.eccentricity)

        data2 = [dataC, dataU]

        axs[0].boxplot(data1, showmeans=True, labels=['Coupled', 'Uncoupled'])
        axs[1].boxplot(data2, showmeans=True, labels=['Coupled', 'Uncoupled'])

        pyplot.title('Eccentricity')
        axs[0].set_title('Channel 0')
        axs[1].set_title('Channel 1')

        pyplot.savefig('boxplot_{}'.format(self.filename))
        pyplot.close()

    def write_spots_and_probs(self, prob_write, directory, title, channels):
        """
        Writes informations about couples and single spots
        :param prob_write: list containing lists of information to write about each couple
        :param directory: string containing the path of the output file
        :param title: name of the output excel file as string
        """
        workbook = xlsxwriter.Workbook(os.path.join(directory, title), {'nan_inf_to_errors': True})
        couples = workbook.add_worksheet(name='Couples')
        titles = ['X1', 'Y1', 'X2', 'Y2', 'Distance', 'Coupling probability']

        titles = ['X1', 'Y1', 'Area_1', 'Distance to Neighbor Same Ch_1', 'Distance to Neighbor Other Ch_1', 'Eccentricity_1', 'Max intensity_1', 'Min intensity_1', 'Mean intensity_1', 'Major axis length_1', 'Minor axis length_1', 'Orientation_1', 'Perimeter_1',
                  'X2', 'Y2', 'Area_2', 'Distance to Neighbor Same Ch_2', 'Distance to Neighbor Other Ch_2', 'Eccentricity_2', 'Max intensity_2', 'Min intensity_2', 'Mean intensity_2', 'Major axis length_2', 'Minor axis length_2', 'Orientation_2', 'Perimeter', 'Coupling Distance', 'Coupling probability']


        for t in range(len(titles)):
            couples.write(0, t, titles[t])

        row = 1
        for p_list in prob_write:
            idx_1, idx_2 = p_list[6], p_list[7]
            (p1, s1, (y1, x1), n_peaks1) = self.MPP1_ROI[idx_1]
            (p2, s2, (y2, x2), n_peaks2) = self.MPP2_ROI[idx_2]

            dnn1_1, nn1, angle = self.neighbors[0][idx_1]
            dnn2_1, nn2, angle = self.neighbors[1][idx_1]

            dnn1_2, nn1, angle = self.neighbors[3][idx_2]
            dnn2_2, nn2, angle = self.neighbors[2][idx_2]

            for index in range(len(p_list)):
                datarow =[x1, y1, s1, dnn1_1, dnn2_1, p1.eccentricity,
                         p1.max_intensity, p1.min_intensity, p1.mean_intensity,
                         p1.major_axis_length, p1.minor_axis_length, p1.orientation,
                         p1.perimeter,

                         x2, y2, s2, dnn1_2, dnn2_2, p2.eccentricity,
                         p2.max_intensity, p2.min_intensity, p2.mean_intensity,
                         p2.major_axis_length, p2.minor_axis_length, p2.orientation,
                         p2.perimeter,

                         p_list[4], p_list[5]
                         ]
                for i in range(len(datarow)):
                    couples.write(row, i, datarow[i])
            row += 1

        spots1 = workbook.add_worksheet(name=f"Spots ch{channels[0]}")
        spots1_array = []
        spots2 = workbook.add_worksheet(name=f"Spots ch{channels[1]}")
        spots2_array = []
        titles = ['X1', 'Y1', 'Area', 'Distance to Neighbor Same Ch', 'Distance to Neighbor Other Ch', 'Eccentricity',
                  'Max intensity', 'Min intensity', 'Mean intensity',
                  'Major axis length', 'Minor axis length', 'Orientation',
                  'Perimeter', 'Peaks', 'Coupled', 'Coupling probability']
        
        for t in range(len(titles)):
            spots1.write(0, t, titles[t])
        row = 1
        for (p1, s1, (y1, x1), n_peaks) in self.MPP1_ROI:
            coupled = 0
            coupling_prob = 0
            for xa, ya, xb, yb, dist, p, _, _ in prob_write:
                if (y1, x1) == (ya, xa):
                    coupled = 1
                    coupling_prob = p
            dnn1, nn1, angle = self.neighbors[0][row-1]
            dnn2, nn2, angle = self.neighbors[1][row-1]

            datarow = [x1, y1, s1, dnn1, dnn2, p1.eccentricity,
                       p1.max_intensity, p1.min_intensity, p1.mean_intensity,
                       p1.major_axis_length, p1.minor_axis_length, p1.orientation,
                       p1.perimeter, n_peaks, coupled, coupling_prob]
            
            spots1_array.append([0, *datarow])
            for i in range(len(datarow)):
                spots1.write(row, i, datarow[i])
                
            row += 1

        titles = ['X2', 'Y2', 'Area', 'Distance to Neighbor Same Ch', 'Distance to Neighbor Other Ch', 'Eccentricity',
                  'Max intensity', 'Min intensity', 'Mean intensity',
                  'Major axis length', 'Minor axis length', 'Orientation',
                  'Perimeter', 'Peaks', 'Coupled', 'Coupling probability']
        titles_array = ["Channel", "X", "Y", "Area", "Distance to Neighbor Same Ch", "Distance to Neighbor Other Ch", "Eccentricity",
                  "Max intensity", "Min intensity", "Mean intensity",
                  "Major axis length", "Minor axis length", "Orientation",
                  "Perimeter", "Peaks", "Coupled", "Coupling probability"]
        for t in range(len(titles)):
            spots2.write(0, t, titles[t])
        row = 1

        for p2, s2, (y2, x2), n_peaks in self.MPP2_ROI:
            coupled = 0
            coupling_prob = 0
            for xc, yc, xd, yd, dist, p, _, _ in prob_write:
                if (y2, x2) == (yd, xd):
                    coupled = 1
                    coupling_prob = p
            dnn1, nn1, angle = self.neighbors[3][row-1]
            dnn2, nn2, angle = self.neighbors[2][row-1]
            datarow = [x2, y2, s2, dnn1, dnn2, p2.eccentricity,
                       p2.max_intensity, p2.min_intensity, p2.mean_intensity,
                       p2.major_axis_length, p2.minor_axis_length, p2.orientation,
                       p2.perimeter, n_peaks, coupled, coupling_prob]
            
            spots2_array.append([1, *datarow])
            for i in range(len(datarow)):
                spots2.write(row, i, datarow[i])
            row += 1
        try:
            workbook.close()
        except PermissionError:
            print("Warning: Workbook is open and couldn't be overwritten!")
        return spots1_array, spots2_array, titles_array

    @staticmethod
    def beta_correction(step, nbN):
        """ This function computes the boundary condition
        beta_correction(maxdist / 10, 100);
        """
        valN = 1/(1/nbN)
        N_h = int(valN) + 1

        alpha = numpy.zeros(N_h + 1)
        results = numpy.zeros(N_h + 1)
        for i in range(1, results.size):
            alpha[i] = (i / N_h)
        for i in range(results.size):
            j = 2
            h = alpha[i] + step
            while h <= 1:
                results[i] = results[i] + h * step / (1 - 1 / numpy.pi * numpy.arccos(alpha[i] / h))
                h = alpha[i] + j * step
                j += 1
            results[i] = results[i] * 2 + alpha[i] * alpha[i]
        return results, N_h

    def distance(self, x1, x2, y1, y2):
        """ This function computes the distance between two points
        :param x1: x coordinates of first point
        :param x2: x coordinates of the second point
        :param y1: y coordinates of the first point
        :param y2: y coordinates of the second point
        """
        return numpy.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def boundary_condition(self, h, x1, x2, y1, y2):
        """ This is to compute the boundary condition (eq 13) in supplementary info
        :param h: The nearest boundary
        :param x1: The x coordinate of point 1
        :param y1: The y coordinate of point 1
        :param X: A numpy 1D array of x random coordinates
        :param Y: A numpy 1D array of y random coordinates

        :returns : The boundary condition
        """
        d = self.distance(x1, x2, y1, y2)
        #minimum = numpy.array([min(h, d) for d in dist])
        if d == 0:
            d = 0.0000001
        minimum = min(h, d)

        k = (1 - 1/numpy.pi * numpy.arccos(minimum / d))
        return 1 / k

    def mean_G(self):
        """ This function computes the mean of G in 2D
        """
        mean = numpy.pi * numpy.diff(numpy.array(self.distance_fit) ** 2)
        return mean