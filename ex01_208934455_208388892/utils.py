import sys

import numpy as np
from PIL import Image
from numba import jit, float32
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()
        self.resized_rgb = np.pad(self.resized_rgb, ((1, 1), (1, 1), (0, 0)), constant_values=0.5)

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
            self.seams_rgb = np.pad(self.seams_rgb, ((1, 1), (1, 1), (0, 0)), constant_values=0.5)

        self.h, self.w = self.rgb.shape[:2]

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))
        # We have added one index_matrix instead of using 2.
        self.index_matrix = np.empty((self.h + 2, self.w + 2), dtype=tuple)
        for i in range(self.h + 2):
            for j in range(self.w + 2):
                self.index_matrix[i, j] = (i, j)

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3)
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        image_np = np.array(np_img)
        gray_np = np.dot(image_np, self.gs_weights)
        bordered_gray_np = np.pad(gray_np, ((1, 1), (1, 1), (0, 0)), constant_values=0.5)
        return bordered_gray_np

    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            In order to calculate a gradient of a pixel, only its neighborhood is required.
        """
        gx = np.diff(self.resized_gs, axis=0, append=0)
        gx[gx.shape[0] - 1] = gx[gx.shape[0] - 2]
        gy = np.diff(self.resized_gs, axis=1, append=0)
        gy[:, gy.shape[1] - 1] = gy[:, gy.shape[1] - 2]
        gradient_magnitude = np.sqrt(np.square(gx) + np.square(gy))
        gradient_magnitude *= 255.0 / np.max(gradient_magnitude)
        return gradient_magnitude

    def calc_M(self):
        pass

    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        pass

    def init_mats(self):
        pass

    def update_ref_mat(self):
        pass

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(self.path)

    @staticmethod
    def load_image(img_path):
        return np.asarray(Image.open(img_path)).astype('float32') / 255.0


class ColumnSeamImage(SeamImage):
    """ Column SeamImage.
    This class stores and implements all required data and algorithmics from implementing the "column" version of the seam carving algorithm.
    """

    def __init__(self, *args, **kwargs):
        """ ColumnSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture, but with the additional constraint:
            - A seam must be a column. That is, the set of seams S is simply columns of M.
            - implement forward-looking cost

        Returns:
            A "column" energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            The formula of calculation M is as taught, but with certain terms omitted.
            You might find the function 'np.roll' useful.
        """
        energy_matrix = self.E
        work_matrix = np.zeros_like(energy_matrix)
        work_matrix[0] = energy_matrix[0]
        # Calculating every row using the former one
        for i in range(1, energy_matrix.shape[0]):
            work_matrix[i] = work_matrix[i - 1] + energy_matrix[i]
        work_matrix = work_matrix + abs(np.roll(self.resized_gs, 1, axis=1) - np.roll(self.resized_gs, -1, axis=1))

        return work_matrix

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matric
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) seam backtracking: calculates the actual indices of the seam
            iii) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            iv) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to support:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        # We have used previously_col_idx for updating only the columns located right to the idx_col_to_remove
        previous_col_idx = 0
        for i in range(num_remove):
            last_row = self.M[-1]
            idx_col_to_remove = last_row.argmin()
            if idx_col_to_remove + 2 < self.M.shape[1] and idx_col_to_remove - 2 > 0:
                if idx_col_to_remove >= previous_col_idx:
                    self.cumm_mask[:, idx_col_to_remove + self.seam_balance] = False
                    self.seam_balance = self.seam_balance + 1
                else:
                    self.cumm_mask[:, idx_col_to_remove] = False
                self.update_E(idx_col_to_remove)
                self.update_M(idx_col_to_remove)
                self.resized_rgb = np.delete(self.resized_rgb, idx_col_to_remove, axis=1)
            else:
                self.M[-1, idx_col_to_remove] = sys.maxsize
            previous_col_idx = idx_col_to_remove

        # Coloring seams_rgb considering the column that has false values in cumm_mask
        for i in range(self.cumm_mask.shape[1]):
            if not self.cumm_mask[:, i].all():
                self.seams_rgb[:, i, :] = (1, 0, 0)

    def update_E(self, seam_idx):
        # Updating only the two relevant cols, using their neighbors for the calculation
        self.E = np.delete(self.E, seam_idx, axis=1)
        self.resized_gs = np.delete(self.resized_gs, seam_idx, axis=1)
        gradient_x_calc = np.diff(self.resized_gs[:, seam_idx - 2:seam_idx + 2], axis=0)
        gradient_y_calc = np.diff(self.resized_gs[:, seam_idx - 2:seam_idx + 2], axis=1)
        gradient_x_calc = np.pad(gradient_x_calc, ((1, 0), (0, 0), (0, 0)), constant_values=0.5)
        gradient_y_calc = np.pad(gradient_y_calc, ((0, 0), (1, 0), (0, 0)), constant_values=0.5)
        gradient_x = gradient_x_calc[:, 1:3]
        gradient_y = gradient_y_calc[:, 1:3]
        gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
        gradient_magnitude *= 255.0 / np.max(gradient_magnitude)
        self.E[:, seam_idx:seam_idx + 2] = gradient_magnitude

    def update_M(self, seam_idx):
        # Updating only the two relevant cols, using their neighbors for the calculation
        self.M = np.delete(self.M, seam_idx, axis=1)
        current_E = self.E[:, seam_idx - 2:seam_idx + 2]
        neighbours_matrix = current_E
        for i in range(1, current_E.shape[0]):
            neighbours_matrix[i] = neighbours_matrix[i - 1] + current_E[i]
        neighbours_matrix[:, 1] += abs(neighbours_matrix[:, 2] - neighbours_matrix[:, 0])
        neighbours_matrix[:, 2] += abs(neighbours_matrix[:, 3] - neighbours_matrix[:, 1])
        self.M[:, seam_idx - 1:seam_idx + 1] = neighbours_matrix[:, 1:3]

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.seam_balance = 0
        self.resized_gs = self.gs
        self.resized_gs = np.rot90(self.gs)
        self.E = np.rot90(self.E)
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M()
        self.cumm_mask = np.ones_like(self.resized_gs, dtype=bool)
        self.seams_rgb = np.rot90(self.seams_rgb)
        self.resized_rgb = np.rot90(self.resized_rgb)
        self.seams_removal(num_remove)
        self.resized_rgb = np.rot90(self.resized_rgb, k=-1)
        self.seams_rgb = np.rot90(self.seams_rgb, k=-1)

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    def backtrack_seam(self):
        """ Backtracks a seam for Column Seam Carving method
        """
        return self.M[:, self.M.argmin(axis=1)[-1]]

    def remove_seam(self):
        """ Removes a seam for self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        energy_matrix = self.E
        M = np.zeros_like(energy_matrix)
        M[0] = energy_matrix[0]
        rows, cols = M.shape[0], M.shape[1]
        return self.calc_M_using_numba(energy_matrix, M, rows, cols, self.resized_gs)

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        for i in range(num_remove):
            self.E = self.calc_gradient_magnitude()
            self.M = self.calc_M()
            last_row = self.M[-1]
            idx_in_last_row_to_remove = last_row.argmin()
            seam_to_remove = self.backtrack_seam_for_removing(idx_in_last_row_to_remove)
            self.update_resized_gs(seam_to_remove)
            self.delete_seam_from_resized_rgb(seam_to_remove)
            self.update_idx_matrix(seam_to_remove)
            self.color_seam_in_seam_rbg(seam_to_remove)

    def backtrack_seam_for_removing(self, idx_in_last_row):
        seam_to_remove = [(self.M.shape[0] - 1, idx_in_last_row)]
        i = self.M.shape[0] - 1
        j = idx_in_last_row
        while i > 0:
            # If j is in the right most col, check only top-top and top-left
            if j >= self.resized_gs.shape[1] - 1:
                if self.M[i, j] == self.E[i, j] + self.M[i - 1, j]:
                    seam_to_remove.append((i, j))
                    i = i - 1
                else:
                    seam_to_remove.append((i, j - 1))
                    i = i - 1
                    j = j - 1
            # If j is in the left most col, check only top-top and top-right
            elif j <= 0:
                if self.M[i, j] == self.E[i, j] + self.M[i - 1, j]:
                    seam_to_remove.append((i, j))
                    i = i - 1
                else:
                    seam_to_remove.append((i, j + 1))
                    i = i - 1
                    j = j + 1
            else:
                c_v = abs(self.resized_gs[i, j + 1] - self.resized_gs[i, j - 1])
                c_l = c_v + abs(self.resized_gs[i - 1, j] - self.resized_gs[i, j - 1])
                c_r = c_v + abs(self.resized_gs[i, j + 1] - self.resized_gs[i - 1, j])
                top_left = self.E[i, j] + self.M[i - 1, j - 1] + c_l
                top_right = self.E[i, j] + self.M[i - 1, j + 1] + c_r

                if self.M[i, j] == top_left:
                    seam_to_remove.append((i, j))
                    i = i - 1
                    j = j - 1
                elif self.M[i, j] == top_right:
                    seam_to_remove.append((i, j))
                    i = i - 1
                    j = j + 1
                else:
                    seam_to_remove.append((i, j))
                    i = i - 1

        return seam_to_remove

    def update_resized_gs(self, seam_to_remove):
        for pair in seam_to_remove:
            i = pair[0]
            j = pair[1]
            if j < self.resized_gs.shape[1] - 1:
                self.resized_gs[i, j:self.resized_gs.shape[1] - 2] = self.resized_gs[i,
                                                                     j + 1:self.resized_gs.shape[1] - 1]
        self.resized_gs = np.delete(self.resized_gs, self.resized_gs.shape[1] - 1, axis=1)

    def update_idx_matrix(self, seam_to_remove):
        for pair in seam_to_remove:
            i = pair[0]
            j = pair[1]
            if j < self.index_matrix.shape[1] - 1:
                self.index_matrix[i, j:self.index_matrix.shape[1] - 2] = self.index_matrix[i,
                                                                         j + 1:self.index_matrix.shape[1] - 1]
        self.index_matrix = np.delete(self.index_matrix, self.index_matrix.shape[1] - 1, axis=1)

    def color_seam_in_seam_rbg(self, seam_to_remove):
        for pair in seam_to_remove:
            i = pair[0]
            j = pair[1]
            if 0 < i < self.index_matrix.shape[0] and 0 < j < self.index_matrix.shape[1]:
                pair_to_remove = self.index_matrix[i, j]
            if 0 < i < self.seams_rgb.shape[0] and 0 < j < self.seams_rgb.shape[1]:
                self.seams_rgb[pair_to_remove[0], pair_to_remove[1]] = (1, 0, 0)

    def delete_seam_from_resized_rgb(self, seam_to_remove):
        for pair in seam_to_remove:
            i = pair[0]
            j = pair[1]
            if 0 < i < self.resized_rgb.shape[0] and 0 < j < self.resized_rgb.shape[1]:
                self.resized_rgb[i, j:self.resized_rgb.shape[1] - 2] = self.resized_rgb[i, j + 1:self.resized_rgb.shape[1] - 1]
        self.resized_rgb = np.delete(self.resized_rgb, self.resized_rgb.shape[1] - 1, axis=1)

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.index_matrix = np.empty((self.w + 2, self.h + 2), dtype=tuple)
        for i in range(self.w + 2):
            for j in range(self.h + 2):
                self.index_matrix[i, j] = (i, j)

        self.resized_gs = self.gs
        self.resized_gs = np.rot90(self.gs)
        self.E = np.rot90(self.E)
        self.seams_rgb = np.rot90(self.seams_rgb)
        self.resized_rgb = np.rot90(self.resized_rgb)
        self.seams_removal(num_remove)
        self.resized_rgb = np.rot90(self.resized_rgb, k=-1)
        self.seams_rgb = np.rot90(self.seams_rgb, k=-1)

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """

    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        # we have implemented our own methods instead of this one.
        raise NotImplementedError("TODO: Implement SeamImage.remove_seam")

    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")

    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    @staticmethod
    @jit(nopython=True)
    def calc_M_using_numba(energy_matrix, M, rows, cols, resized_gs):
        for i in range(1, rows):
            for j in range(cols):
                if j == 0:
                    if M[i - 1, j] < M[i - 1, j + 1]:
                        M[i, j] = M[i - 1, j] + energy_matrix[i, j]
                    else:
                        M[i, j] = M[i - 1, j + 1] + energy_matrix[i, j]
                elif j == cols - 1:
                    if M[- 1, j - 1] < M[i - 1, j]:
                        M[i, j] = M[i - 1, j - 1] + energy_matrix[i, j]
                    else:
                        M[i, j] = M[i - 1, j] + energy_matrix[i, j]
                else:
                    top_left = M[i - 1, j - 1] + np.abs(resized_gs[i, j + 1] - resized_gs[i, j - 1]) + np.abs(
                        resized_gs[i - 1, j] - resized_gs[i, j - 1])
                    top_top = M[i - 1, j] + np.abs(resized_gs[i, j + 1] - resized_gs[i, j - 1])
                    top_right = M[i - 1, j + 1] + np.abs(resized_gs[i, j + 1] - resized_gs[i, j - 1]) + np.abs(
                        resized_gs[i, j + 1] - resized_gs[i - 1, j])

                    if top_left <= top_top and top_left <= top_right:
                        M[i, j] = top_left + energy_matrix[i, j]
                    elif top_right <= top_top and top_right <= top_left:
                        M[i, j] = top_right + energy_matrix[i, j]
                    else:
                        M[i, j] = top_top + energy_matrix[i, j]

        return M


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """

    return np.multiply(orig_shape, scale_factors).astype(int)


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    seam_img.reinit()
    num_of_vertical_seams_to_remove = shapes[0][1] - shapes[1][1]
    num_of_horizontal_seams_to_remove = shapes[0][0] - shapes[1][0]

    seam_img.seams_removal_vertical(num_of_vertical_seams_to_remove)
    seam_img.seams_removal_horizontal(num_of_horizontal_seams_to_remove)

    return seam_img.resized_rgb


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x, in_width, out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y, in_height, out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:, x1s] * dx + (1 - dx) * image[y1s][:, x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:, x1s] * dx + (1 - dx) * image[y2s][:, x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image
