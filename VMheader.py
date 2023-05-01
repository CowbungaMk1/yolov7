from PIL import Image
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import spatial
import math
from scipy.signal import argrelextrema
import openturns as ot


def deg_to_dms(deg):  # degrees, minutes, seconds. Returns string
    d = int(deg)
    md = abs(deg - d) * 60
    m = float(md)
    sd = (md - m) * 60
    return str(d) + str(m)


def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    # assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def matching(vector_drone, vector_sat_all, features_req, idx):
    # marked for revision
    vector_sat = []
    kk = []

    for k in range(len(vector_sat_all)):

        if vector_drone[idx][0:features_req * 4] == vector_sat_all[k][0:features_req * 4]:
            # compair drone to sat, keep only features of the same classes
            vector_sat.append(vector_sat_all[k])
            kk.append(k)

    H = []
    VV = []  # storing all hamming distances between all drone and all satellite features

    for j in range(len(vector_sat)):
        H.append(hamming2(vector_sat[j], vector_drone[idx]))  # using hamming distance
        VV.append(vector_sat[j])

    if len(H) != 0:
        II = np.where(H == np.sort(H)[0])[0]  # index of the lowest hamming distance
        return II, kk
    else:
        II = []
        return II, kk


# finding local minima of KDE
# previously, bands = [ .4 for angle, .1 for brightness and .1 for dist]
def find_mins(im_dist, bandwidth):  # same but for distances

    R = np.sort(np.array(im_dist).flatten())
    im_dist = np.reshape(R, (len(R), 1))
    # print(im_dist)
    x_d = np.linspace(0, max(im_dist)[0], 100)
    # print(R)
    sample = ot.Sample([[hh] for hh in R])
    factory = ot.KernelSmoothing()

    band = bandwidth * factory.computePluginBandwidth(sample)[0]

    kde = KernelDensity(kernel='gaussian', bandwidth=band)
    kde.fit(im_dist)
    prob = np.exp(kde.score_samples(x_d[:, None]))
    min_ind_dist = [0, *argrelextrema(prob, np.less)[0]]
    min_ind_dist.insert(len(min_ind_dist), len(prob))

    xx_d = x_d  # sems redundant and is completed in bin_dist
    x_d = [*x_d, max(im_dist)[0] + 1]

    return min_ind_dist, prob, x_d, im_dist, xx_d


# binning() takes the measurement desired to be placed into a bin, the lines pace of 100 values between 0 and max
# index of x_d of local minima to divide the bins [min_ind_dist]
def binning(D, min_ind_dist, x_d):  # number of bins, see fig 13
    H = []

    for j in range(len(min_ind_dist) - 1):
        if x_d[min_ind_dist[j]] <= D <= x_d[min_ind_dist[j + 1]]:
            H.append('{0:04b}'.format(j))  # 'bin' + str(j)
    if len(H) == 0:
        H.append('{0:04b}'.format(15))
    return H  # vectors of bin numbers in binary and using 8 digits


# Data collection from detected features
class VectorFormatter:

    def __init__(self, name):
        self.name = name

        self.features = []
        self.feature_class = []

        self.pos_x = []
        self.pos_y = []
        self.p_x = []
        self.p_y = []
        self.d_x = []
        self.d_y = []
        self.LLL = []
        self.brightness = []
        self.x_humvee = []
        self.y_humvee = []

        self.im_angle = []
        self.im_br = []
        self.im_dist = []
        self.pos_x_closest_5_a = []
        self.pos_y_closest_5_a = []
        self.base_x = []
        self.base_y = []
        self.feature_closest = []
        self.true_match = []
        self.humvee_detected = False

    # For detector derived data
    def cnn_initialization(self, results, dimensions):

        self.pos_x = []
        self.pos_y = []
        self.shape_y = dimensions[0]
        self.shape_x = dimensions[1]

        self.feature_class = []

        for i in range(len(results)):

            self.features.append(results[i])
            V = results[i]

            if V[0] == float(0):  # Building
                self.feature_class.append(0.0)
            if V[0] == float(1):  # 'Water':
                self.feature_class.append(1.0)
            if V[0] == float(2):  # 'Intersection':
                self.feature_class.append(2.0)
            if V[0] == float(3):  # 'Rock Pile':
                self.feature_class.append(3.0)
            if V[0] == float(4):  # 'Tree Clump':
                self.feature_class.append(4.0)
            if V[0] == float(5):  # 'Tree Opening':
                self.feature_class.append(5.0)
            if V[0] == float(6):  # 'Humvee':
                self.feature_class.append(6.0)
                self.humvee_detected = True

            x_min = V[1]  # Reading from detection output, Finding objects x and y dimensions
            y_min = V[2]
            x_max = V[3]
            y_max = V[4]

            center_x = (x_min + (x_max - x_min) / 2) / self.shape_x
            center_y = (y_min + (y_max - y_min) / 2) / self.shape_y

            if V[0] == float(6):
                self.x_humvee.append(center_x)
                self.y_humvee.append(center_y)

            # photo = photo.convert('RGB')  # need to calculate brightness of centroid, must be converted to RGB first
            # # coordinates of the pixel

            # X, Y = int(center_x * self.shape_x), int(center_y * self.shape_y)
            # # Get RGB
            # pixelRGB = photo.getpixel((X, Y))
            # R, G, B = pixelRGB
            # self.brightness.append(sum([R, G, B]) / 3)  # 0 is dark (black) and 255 is bright (white)

            self.pos_x.append(center_x)  # Center in percents
            self.pos_y.append(center_y)

            self.p_x.append(center_x * self.shape_x)  # This the center in pixel coordinates
            self.p_y.append(center_y * self.shape_y)

    # For manually labeled data
    def yolo_initialize(self, file_txt, image_file):

        im = Image.open(image_file)
        # self.shape_y = im.height
        self.shape_x = im.width
        self.shape_y = im.height
        with open(file_txt) as f:
            AA = f.readlines()

            for i in range(len(AA)):
                self.features.append(AA[i])

                V = AA[i].split(' ')
                self.feature_class.append(float(V[0]))  # uses indices of classes not the string name

                if V[0] == '6':
                    self.x_humvee.append(float(V[1]))
                    self.y_humvee.append(float(V[2]))

                imag = im.convert('RGB')

                # coordinates of the pixel

                self.pos_x.append(float(V[1]))
                self.pos_y.append(float(V[2]))

                self.p_x.append(float(V[1]) * self.shape_x)
                self.p_y.append(float(V[2]) * self.shape_y)

                if len(V) > 5:
                    self.true_match.append(V[5].split(('\n'))[0])

    def k_d_tree_test(self, num_neighbors):

        tree = spatial.KDTree(list(zip(self.pos_x, self.pos_y)))  # KDTree positions of features detected

        for ii in range(len(self.features)):
            pts = np.array([self.pos_x[ii], self.pos_y[ii]])  # Coordinates of the point r in fig 5 in paper

            dist_to_neighbor, idx_neighbor = tree.query(pts, k=num_neighbors)

            # print(dist_to_neighbor)
            dist_to_neighbor = dist_to_neighbor[1:]  # removing trivial case, ie its "nearest" point is itself
            idx_neighbor = idx_neighbor[1:]
            #
            norm_term = dist_to_neighbor[0]  # Normalize the distance values with the nearest neighbor
            if norm_term != 0:
                dist_to_neighbor = [x / norm_term for x in dist_to_neighbor]

            dist_to_neighbor = dist_to_neighbor[
                               1:]  # removing 2nd trivial case, After normalization first index is always one

            self.im_dist.append(dist_to_neighbor)
            # self.im_br.append(br)
            pos_x_closest = []
            pos_y_closest = []

            pos_x_base = self.pos_x[idx_neighbor[0]]  # sets up base position
            pos_y_base = self.pos_y[idx_neighbor[0]]

            dist_to_neighbor = dist_to_neighbor[1:]  # removing the "base" , the focrum of the angle
            idx_neighbor = idx_neighbor[1:]
            alpha = []
            dist = []
            br = []

            j = 0
            feature_closest_list = []
            for j in range(len(idx_neighbor)):
                pos_x_closest.append(self.pos_x[idx_neighbor[j]])
                pos_y_closest.append(self.pos_y[idx_neighbor[j]])

                feature_closest_list.append(self.feature_class[idx_neighbor[j]])

            self.feature_closest.append(feature_closest_list)

            # computing distance between p and m, and the angle alpha
            for i in range(len(idx_neighbor)):
                a = [pos_x_base - pts[0], pos_y_base - pts[1]]
                b = [pos_x_closest[i] - pts[0], pos_y_closest[i] - pts[1]]
                alpha.append(math.atan2(np.linalg.norm(np.cross(a, b)),
                                        np.dot(a, b)))  # calculating angle between base and other neighbors
            alpha_deg = []
            ang = []

            for i in range(len(alpha)):
                alpha_deg.append(np.degrees(alpha[i]))
                ang.append(int(np.degrees(alpha[i])))

            # self.im_dist.append(dist)
            self.im_br.append(br)
            self.im_angle.append(alpha)

    # Class variables
    bandwidth = 0.005


# defining vector in binary
class BinCompletion:
    def __init__(self, name):

        self.name = name
        self.min_ind_angle = []
        self.min_ind_dist = []
        self.min_ind_brightness = []
        self.x_d_angle = []
        self.x_d = []
        self.x_d_brightness = []

    def bin_initialize(self, indiv_dist, indiv_angle, bw_dist, bw_angle):
        self.min_ind_dist, prob, self.x_d, im_dist, xx_d = find_mins(indiv_dist, bw_dist)
        self.min_ind_angle, prob_angle, self.x_d_angle, im_angle, xx_d_angle = find_mins(indiv_angle, bw_angle)

    def vector_def(self, num, feature_class, indiv_dist, indiv_angle, features_closest):
        Vector = []
        feature = feature_class[num]
        Vector.append('{0:04b}'.format(int(feature)))  # first four digits represent feature class

        for j in range(len(features_closest[num]) - 1):
            Vector.append('{0:04b}'.format(int(features_closest[num][j])))

        for j in range(len(features_closest[num])):
            D = indiv_dist[num][j]
            H = binning(D, self.min_ind_dist, self.x_d)
            Vector.append(H[0])

        for j in range(len(features_closest[num])):
            D_angle = indiv_angle[num][j]
            H_angle = binning(D_angle, self.min_ind_angle, self.x_d_angle)
            Vector.append(H_angle[0])

        Vector = ''.join(Vector)
        return Vector, []
