# -*- coding:utf-8 -*-
import time
import cv2 as cv
from skimage.color import rgb2gray
from scipy import misc
from skimage import img_as_ubyte
from skimage.draw import polygon_perimeter
from skimage.filters import threshold_otsu
import numpy as np 
from matplotlib import pyplot as plt 

ADDRESS = 'image/qr3.PNG'
nan = float('inf')
DEBUG = True

def check_ratio(state_count) -> bool:
    total_finder_size = 0
    for i in range(5):
        # if any state count is zero this is not a finder pattern
        if state_count[i] == 0:
            return False
        total_finder_size += state_count[i]

    if total_finder_size < 7:
        return False

    # calculate the size of one module
    module_size = total_finder_size / 7
    max_variance = module_size / 2
    # check the ratio 1:1:3:1:1
    # we need to allow for quite a large variance because of the way the image may be tilted
    return_val = ((abs(module_size - state_count[0]) < max_variance) and
                  (abs(module_size - (state_count[1])) < max_variance) and
                  (abs(3 * module_size - state_count[2]) < 3 * max_variance) and
                  (abs(module_size - (state_count[3])) < max_variance) and
                  (abs(module_size - (state_count[4])) < max_variance))

    return return_val  

def center_from_end(state_count: np.ndarray, col: int) -> float:
    return col - state_count[4] - state_count[3] - (state_count[2] / 2.0)


def compute_dimension(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, module_size: float) -> int:
    # the dimension is always a square of 21 x 21, 25 x 25, 29 x29, etc.
    diff_top = tl - tr
    diff_left = tl - bl

    # calculate the distance between top-* and *-left points
    dist_top = np.sqrt(diff_top.dot(diff_top))
    dist_left = np.sqrt(diff_left.dot(diff_left))

    # calculate the number of modules
    width = round(dist_top / module_size)
    height = round(dist_left / module_size)
    dimension = 7 + (width + height) / 2

    # the dimension must have a remainder of 4, so we need to enforce this
    rem = dimension % 4
    if rem == 0:
        dimension += 1
    elif rem == 2:
        dimension -= 1
    elif rem == 3:
        dimension -= 2

    return dimension

class QRcode:
    def __init__(self, im: np.ndarray, row_increment=3, black_background=True):
        self.MIN_DIST_THRESHOLD = 10
        self.black_background = black_background
        self.found = False
        self.row_increment = row_increment
        self.possible_centers = []
        self.module_size = []
        self.orig_image = im
        if im.shape[2] > 2:
            # covert to grayscale
            self.image = rgb2gray(im)
        else:
            self.image = im

        # threshold to binary
        thres = threshold_otsu(self.image)
        if black_background:
            self.image = self.image < thres
        else:
            self.image = self.image > thres
        self.image = img_as_ubyte(self.image)

    def sort_finder_patterns(self):
        # the finder patterns may be in any order
        # we need them to be in this order [tl, tr, bl] for further processing
        # first calculate the distances between the three points
        distances = [np.linalg.norm(self.possible_centers[0] - self.possible_centers[1]),
                     np.linalg.norm(self.possible_centers[0] - self.possible_centers[2]),
                     np.linalg.norm(self.possible_centers[1] - self.possible_centers[2])]
        max_dist = np.argmax(distances)

        # the longest distance indicates which finder is in the top-left corner
        # now we need to determine which is the bl. The sign of the cross product between the
        # distance vectors (tl - bl) and (tl - tr) will determine which is which (right-hand rule)
        if max_dist == 0:
            tl = self.possible_centers[2]
            cross = np.cross(self.possible_centers[0] - self.possible_centers[2],
                             self.possible_centers[1] - self.possible_centers[2])
            if cross > 0:
                bl = self.possible_centers[0]
                tr = self.possible_centers[1]
            else:
                bl = self.possible_centers[1]
                tr = self.possible_centers[0]
        elif max_dist == 1:
            tl = self.possible_centers[1]
            cross = np.cross(self.possible_centers[0] - self.possible_centers[1],
                             self.possible_centers[2] - self.possible_centers[1])
            if cross > 0:
                bl = self.possible_centers[0]
                tr = self.possible_centers[2]
            else:
                bl = self.possible_centers[2]
                tr = self.possible_centers[0]
        else:
            tl = self.possible_centers[0]
            cross = np.cross(self.possible_centers[1] - self.possible_centers[0],
                             self.possible_centers[2] - self.possible_centers[0])
            if cross > 0:
                bl = self.possible_centers[1]
                tr = self.possible_centers[2]
            else:
                bl = self.possible_centers[2]
                tr = self.possible_centers[1]

        self.possible_centers = np.asarray([tl, tr, bl])

    def find_alignment_marker(self, img: np.ndarray) -> bool:
        # check the number of finder patterns to makes sure we have a valid qr code
        if len(self.possible_centers) != 3:
            return False
        self.sort_finder_patterns()

    def handle_possible_center(self, state_count: np.ndarray, row: int, col: int) -> bool:
        state_count_total = int(np.sum(state_count))

        # if DEBUG and 191 > row > 185:
        #     print('row: '.format(row))

        # cross check along the vertical axis
        center_col = center_from_end(state_count, col)
        center_row = self.cross_check_vertical(row, int(center_col), state_count[2], state_count_total)
        if center_row == nan:
            return False

        # cross check along the horizontal axis with new center row
        center_col = self.cross_check_horizontal(int(center_row), int(center_col), state_count[2], state_count_total)
        if center_col == nan:
            return False

        # cross check along the diagonal with new center row and column
        valid_finder = self.cross_check_diagonal(int(center_row), int(center_col), state_count[2], state_count_total)
        if valid_finder == nan:
            return False

        # check for duplicates
        new_finder_center = np.asarray([center_row, center_col])
        new_estimated_module_size = state_count_total / 7.0
        found = False
        idx = 0

        for i in range(len(self.possible_centers)):
            diff = self.possible_centers[i] - new_finder_center
            dist = np.sqrt(np.dot(diff, diff))

            if dist < self.MIN_DIST_THRESHOLD:
                # these two centers are very close and likely the same
                # improve the center estimate by taking the mean
                self.possible_centers[i] = (self.possible_centers[i] + new_finder_center) / 2
                self.module_size[idx] = (self.module_size[idx] + new_estimated_module_size) / 2
                found = True
                break

        if not found:
            # this must be a new center
            self.module_size.append(new_estimated_module_size)
            self.possible_centers.append(new_finder_center)
            return True

        return False

    def find_qr(self) -> bool:
        for row in range(self.row_increment, self.image.shape[0], self.row_increment):
            state_count = np.zeros((5,))
            current_state = 0
            for i, px in enumerate(self.image[row, :]):
                if px == 0:
                    # this is a black pixel
                    if current_state == 1 or current_state == 3:
                        # we were counting white pixels so the state needs to change from white to black
                        current_state += 1
                    # increment the state count
                    state_count[current_state] += 1
                else:
                    # this is a white pixel
                    if current_state == 1 or current_state == 3:
                        # remain in the white pixel state we are still counting white pixels
                        state_count[current_state] += 1
                    else:
                        # but we were counting black pixels
                        if current_state == 4:
                            # we found the white border after the finder pattern
                            # so now check whether this is the correct ratio 1:1:3:1:1
                            if check_ratio(state_count) and self.handle_possible_center(state_count, row, i):
                                # we have a match
                                if DEBUG:
                                    print('found center at {}:{}'.format(
                                        self.possible_centers[-1][0], self.possible_centers[-1][1]))
                                    print('module size: {}'.format(self.module_size[-1]))
                            else:
                                # this is not a valid code so shift the state machine
                                current_state = 3
                                state_count[0] = state_count[2]
                                state_count[1] = state_count[3]
                                state_count[2] = state_count[4]
                                state_count[3] = 1
                                state_count[4] = 0
                                continue
                            # reset the state machine to start looking for more matches
                            current_state = 0
                            state_count[0] = 0
                            state_count[1] = 0
                            state_count[2] = 0
                            state_count[3] = 0
                            state_count[4] = 0
                        else:
                            # we are still in the pattern so increment the state
                            current_state += 1
                            state_count[current_state] += 1

        return len(self.possible_centers) == 3

    def show_finders(self) -> np.ndarray:
        im = self.orig_image.copy()
        if len(im.shape) < 3 or im.shape[2] == 1:
            col = 255
            shp = im.shape
        elif im.shape[2] == 4:
            col = [255, 0, 0, 255]
            shp = im.shape[:2]
        elif im.shape[2] == 3:
            col = [255, 0, 0]
            shp = im.shape[:2]
        else:
            print('WARNING: Could not determine colour channel format for image')
            return im

        for i in range(len(self.possible_centers)):
            offset = self.module_size[i] * 3.5
            center = self.possible_centers[i]
            row_coordinates = [center[0] - offset, center[0] - offset, center[0] + offset, center[0] + offset]
            col_coordinates = [center[1] - offset, center[1] + offset, center[1] + offset, center[1] - offset]
            rr, cc = polygon_perimeter(row_coordinates, col_coordinates, shape=shp)
            im[rr, cc] = col

        return im

    def cross_check_vertical(self, start_row: int, center_col: int, central_count: int,
                             state_count_total: int) -> float:
        max_rows = self.image.shape[0]
        cross_check_state_count = np.zeros((5,))
        row = start_row

        while row >= 0 and self.image[row, center_col] == 0:
            # walk upwards from the center of the finder
            cross_check_state_count[2] += 1
            row -= 1
        if row < 0:
            return nan

        while row >= 0 and self.image[row, center_col] == 255 and cross_check_state_count[1] < central_count:
            # walk upwards from the first border of the finder
            cross_check_state_count[1] += 1
            row -= 1
        if row < 0 or cross_check_state_count[1] >= central_count:
            return nan

        while row >= 0 and self.image[row, center_col] == 0 and cross_check_state_count[0] < central_count:
            # walk upwards from the second border of the finder
            cross_check_state_count[0] += 1
            row -= 1
        if row < 0 or cross_check_state_count[0] >= central_count:
            return nan

        # now we traverse down the center
        row = start_row + 1
        while row < max_rows and self.image[row, center_col] == 0:
            # walk down from the center of the finder
            cross_check_state_count[2] += 1
            row += 1
        if row == max_rows:
            return nan

        while row < max_rows and self.image[row, center_col] == 255 and cross_check_state_count[3] < central_count:
            # walk down from the first border of the finder
            cross_check_state_count[3] += 1
            row += 1
        if row == max_rows or cross_check_state_count[3] >= central_count:
            return nan

        while row < max_rows and self.image[row, center_col] == 0 and cross_check_state_count[4] < central_count:
            # walk down from the second border of the finder
            cross_check_state_count[4] += 1
            row += 1
        if row == max_rows or cross_check_state_count[4] >= central_count:
            return nan

        # finally check to make sure the ration 1:1:3:1:1 holds
        cross_check_state_count_total = int(np.sum(cross_check_state_count))

        # check that the state count is similar to the original state count (this bit is confusing in the tutorial)
        if 5 * np.abs(cross_check_state_count_total - state_count_total) >= 2 * state_count_total:
            return nan

        # check the ratio
        center = center_from_end(cross_check_state_count, row)
        if check_ratio(cross_check_state_count):
            return center
        else:
            return nan

    def cross_check_horizontal(self, center_row: int, start_col: int, central_count: int,
                               state_count_total: int) -> float:
        max_cols = self.image.shape[1]
        cross_check_state_count = np.zeros((5,))
        col = start_col

        while col >= 0 and self.image[center_row, col] == 0:
            # walk left from the center of the finder
            cross_check_state_count[2] += 1
            col -= 1
        if col < 0:
            return nan

        while col >= 0 and self.image[center_row, col] == 255 and cross_check_state_count[1] < central_count:
            # walk left from the first border of the finder
            cross_check_state_count[1] += 1
            col -= 1
        if col < 0 or cross_check_state_count[1] >= central_count:
            return nan

        while col >= 0 and self.image[center_row, col] == 0 and cross_check_state_count[0] < central_count:
            # walk left from the second border of the finder
            cross_check_state_count[0] += 1
            col -= 1
        if col < 0 or cross_check_state_count[0] >= central_count:
            return nan

        # now we traverse right from the center
        col = start_col + 1
        while col < max_cols and self.image[center_row, col] == 0:
            # walk right from the center of the finder
            cross_check_state_count[2] += 1
            col += 1
        if col == max_cols:
            return nan

        while col < max_cols and self.image[center_row, col] == 255 and cross_check_state_count[3] < central_count:
            # walk right from the first border of the finder
            cross_check_state_count[3] += 1
            col += 1
        if col == max_cols or cross_check_state_count[3] >= central_count:
            return nan

        while col < max_cols and self.image[center_row, col] == 0 and cross_check_state_count[4] < central_count:
            # walk right from the second border of the finder
            cross_check_state_count[4] += 1
            col += 1
        if col == max_cols or cross_check_state_count[4] >= central_count:
            return nan

        # finally check to make sure the ration 1:1:3:1:1 holds
        new_state_count_total = int(np.sum(cross_check_state_count))

        # check that the state count is similar to the original state count (this bit is confusing in the tutorial)
        if 5 * np.abs(new_state_count_total - state_count_total) >= state_count_total:
            return nan

        # check the ratio
        center = center_from_end(cross_check_state_count, col)
        if check_ratio(cross_check_state_count):
            return center
        else:
            return nan

    def cross_check_diagonal(self, center_row: int, center_col: int, max_count: int, state_count_total: int) -> bool:
        cross_check_state_count = np.zeros((5,))

        i = 0
        while center_row >= i and center_col >= i and self.image[center_row - i, center_col - i] == 0:
            # walk to the top-left from the center of the finder
            cross_check_state_count[2] += 1
            i += 1
        if center_row < i or center_col < i:
            return False

        while center_row >= i and center_col >= i and self.image[center_row - i, center_col - i] == 255 \
                and cross_check_state_count[1] <= max_count:
            # walk to the top-left from the first border of the finder
            cross_check_state_count[1] += 1
            i += 1
        if center_row < i or center_col < i or cross_check_state_count[1] > max_count:
            return False

        while center_row >= i and center_col >= i and self.image[center_row - i, center_col - i] == 0 \
                and cross_check_state_count[0] <= max_count:
            # walk to the top-left from the second border of the finder
            cross_check_state_count[0] += 1
            i += 1
        if cross_check_state_count[1] > max_count:
            return False

        # now we traverse to the bottom-right from the center
        max_cols = self.image.shape[1]
        max_rows = self.image.shape[0]
        i = 1
        while center_row + i < max_rows and center_col + i < max_cols and self.image[
            center_row + i, center_col + i] == 0:
            # walk to the bottom-right from the center of the finder
            cross_check_state_count[2] += 1
            i += 1
        if center_row + i >= max_rows or center_col + i >= max_cols:
            return False

        while center_row + i < max_rows and center_col + i < max_cols and self.image[
            center_row + i, center_col + i] == 255 and cross_check_state_count[3] < max_count:
            # walk to the bottom-right from the first border of the finder
            cross_check_state_count[3] += 1
            i += 1
        if center_row + i >= max_rows or center_col + i >= max_cols or cross_check_state_count[3] > max_count:
            return False

        while center_row + i < max_rows and center_col + i < max_cols and self.image[
            center_row + i, center_col + i] == 0 and cross_check_state_count[4] < max_count:
            # walk to the bottom-right from the second border of the finder
            cross_check_state_count[4] += 1
            i += 1
        if center_row + i >= max_rows or center_col + i >= max_cols or cross_check_state_count[4] > max_count:
            return False

        # finally check to make sure the ration 1:1:3:1:1 holds
        new_state_count_total = int(np.sum(cross_check_state_count))

        # check that the state count is similar to the original state count (this bit is confusing in the tutorial)
        return np.abs(new_state_count_total - state_count_total) < 2 * state_count_total \
               and check_ratio(cross_check_state_count)

def show (img):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.imshow(img)
    fig.show()

def preprocess (img, code = cv.COLOR_BGR2GRAY):
    img_gray = cv.cvtColor(img, code)
    img_gb = cv.GaussianBlur(img_gray, (5,5), 0)
    img_th = cv.adaptiveThreshold(img_gb, 255, 
                                  cv.ADAPTIVE_THRESH_MEAN_C, 
                                  cv.THRESH_BINARY, 5, 0)
    '''用Canny会造成识别轮廓重合度高'''
    img_canny = cv.Canny(img_th, 100, 200)
    return img_th

if __name__ == '__main__':
    image = misc.imread(ADDRESS, mode = 'RGB')
    q_reader = QRcode(image, black_background = True)
    start = time.clock()
    print('qr found: {}'.format(q_reader.find_qr()))
    end = time.clock()
    print ('cost time : {}'.format(end - start))
    finders = q_reader.show_finders()
    if DEBUG:
        plt.imshow(finders)
        plt.show()
