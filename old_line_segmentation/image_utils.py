"""
U-Net segmentation / postprocessing code
Only Jonas really knows what this does
"""

from statistics import median, mode, multimode
from typing import Dict, Tuple

import cv2
import numpy as np
import skimage
from scipy.signal import argrelextrema
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KernelDensity


def return_largest_value(_list: list) -> float:
    """
    Returns the largest element in a list containing only integers or floats
    """
    # check that all elements in {_list} are digits
    if all([any([isinstance(l, int), isinstance(l, float)]) for l in _list]):
        _list = sorted(_list, reverse=True)
        return _list[0]
    else:
        raise ValueError("Not all elements in the list are digits")


def draw_rectangle(arr, top_left, bottom_right, thickness=1, value=1):
    """
    Draw a hollow rectangle (box) in a numpy array with specified thickness.

    Parameters:
        arr (numpy.ndarray): The numpy array to draw the rectangle on (shape h, w(, c).
        top_left (tuple): Coordinates of the top-left corner of the rectangle (row, column).
        bottom_right (tuple): Coordinates of the bottom-right corner of the rectangle (row, column).
        thickness (int): Thickness of the rectangle border (default is 1).
        value (int): Value to fill the rectangle border with (default is 1).
    """
    arr = arr.copy()
    # Draw top and bottom borders
    arr[top_left[0]:top_left[0] + thickness, top_left[1]:bottom_right[1] + 1] = value
    arr[bottom_right[0] - thickness + 1:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1] = value

    # Draw left and right borders
    arr[top_left[0]:bottom_right[0] + 1, top_left[1]:top_left[1] + thickness] = value
    arr[top_left[0]:bottom_right[0] + 1, bottom_right[1] - thickness + 1:bottom_right[1] + 1] = value

    return arr


def illustrate_layout(img: np.array, coords: Dict[str, Tuple[int]]) -> np.ndarray:
    """
    Draws boxes around detected (birthday) cells for easy validation
    """
    # Define color space
    colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
    }
    # Set counter for circling {colors}

    # for counter, (_, c) in enumerate(coords.items()):
    #     cv2.rectangle(cv2.UMat(img), (c[2], c[0]), (c[3], c[1]), color=colors[counter % 3], thickness=5)

    for counter, (_, c) in enumerate(coords.items()):
        img = draw_rectangle(img, (c[0], c[2]), (c[1], c[3]), thickness=3, value=colors[counter % 3])
    return img


class ImageSegmentation:
    def __init__(self, config: dict, preloaded_img, preloaded_seg):
        self.config = config
        self.illustration_dir = config["illustration_dir"]
        self.cells_dir = config["cells_dir"]
        self.unet_pred_dir = config["unet_pred_dir"]

        self.img = preloaded_img.astype("uint8")
        self.seg = preloaded_seg
        self.HEIGHT = self.img.shape[0]
        self.WIDTH = self.img.shape[1]
        self.IDEAL_NUMBER_OF_COLUMNS = 8  # CHANGED: 6
        self.minD = 0.05 * self.HEIGHT  # minimum distance
        self.skeletonized_mask = None
        self.columns = None
        self.complete_columns = None
        self.mode_number_rows = None
        self.corners = None
        self.interpolated_corners = []

    def list_nearest_from_list(self, l1: list, l2: list) -> list:
        """
        Returns the nearest item in {l1} for every item in {l2}
        Example
            Input
                l1 = [1, 4, 11, 20, 25]; l2 = [3, 10, 20]
            Returns
                [4, 11, 20]
        """
        return list(map(lambda y: min(l1, key=lambda x: abs(x - y)), l2))

    def find_first_white_pixel(self, direction: str, invert: bool = False) -> int:
        """
        Returns index of first non-zero pixel from direction={direction}.
        """
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if invert:
            img = 255 - img
        if direction == "left":
            # look from the left for first white pixel mid image
            for j in range(self.WIDTH):
                if img[self.HEIGHT // 2, j] == 255:
                    return j
        elif direction == "top":
            for j in range(self.HEIGHT):
                if img[j, self.WIDTH // 4] == 255:
                    return j
        elif direction == "bottom":
            for j in range(self.HEIGHT):
                if img[-j, self.WIDTH // 4] == 255:
                    return j
        else:
            raise ValueError("Invalid {direction}: " + f"{direction}")

    def check_alignment(self, c1: np.ndarray, c2: np.ndarray, also_vertical: bool = True, m: int = 10) -> bool:
        """
        Checks if coordinates {c1, c2} are pairwise aligned both
        horizontally and vertically within acceptable margin {m}
        Args:
            c1: array of coordinates
            c2: array of coordinates
            also_vertical: whether to also check for vertical alignment
            m: int, degrees for margin
        Returns:
            True if all coordinate pairs are aligned, False otherwise
        """
        # check horizontal alignment
        for i in range(len(c1)):
            dx = c2[i, 1] - c1[i, 1]
            dy = c2[i, 0] - c1[i, 0]
            if dy == 0:
                continue
            else:
                slope = np.abs(np.rad2deg(np.arctan2(dy, dx)))
                if (0 - m < slope) & (slope < 0 + m):
                    continue
                else:
                    # print(f'Not horizontally aligned at idx={i}')
                    return False

        if also_vertical:
            # check vertical alignment, but only for {c1}
            for e, i in enumerate(range(1, len(c1))):
                if e == len(c1) - 2:  # last row
                    continue
                else:
                    dx = c1[i + 1, 1] - c1[i, 1]
                    dy = c1[i + 1, 0] - c1[i, 0]
                    if dx == 0:
                        continue
                    else:
                        slope = np.abs(np.rad2deg(np.arctan2(dy, dx)))
                        if (90 - m < slope) & (slope < 90 + m):
                            continue
                        else:
                            # print('Not vertically aligned')
                            return False
        return True

    def clear_top_and_bottom(self):
        """
        Removes corners from top and bottom, determined by looking for the first
        non-black pixel from both directions
        """
        # locate black frame at top and bottom
        black_top = self.find_first_white_pixel(direction="top")
        black_bottom = self.find_first_white_pixel(direction="bottom")
        # remove corners detected at header / footer
        self.corners = np.delete(
            self.corners,
            np.argwhere(
                (self.corners[:, 0] < 0.1 * self.HEIGHT + black_top)
                | (self.corners[:, 0] > 0.95 * self.HEIGHT - black_bottom)
            ),
            axis=0,
        )
        return

    def get_columns(self) -> list:
        """
        Uses a kernel density estimator to group {self.corners} into columns
        Returns:
            Coordinates in a list of arrays by column
        """
        self.clear_top_and_bottom()
        c = self.corners[:, 1]  # the x coordinates

        # group coordinates using a kernel density estimator
        # NOTE: experiment more
        kde = KernelDensity(bandwidth=1.0, kernel="exponential").fit(c.reshape(-1, 1))
        s = np.linspace(np.min(c), np.max(c))
        e = kde.score_samples(s.reshape(-1, 1))
        # find minima and maxima
        mi, _ = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

        # identify columns
        cols = []  # list to store columns
        col0 = self.corners[c < s[mi[0]]]
        if len(col0) > 1:  # check for at least two corners in a row
            cols.append(col0)
        idx = 0  # set index for column in while-loop
        while len(cols) < self.IDEAL_NUMBER_OF_COLUMNS:  # CHANGED: -1:
            # NOTE: by lowering the number of columns it should return now,
            # we force it to do column search later. This is only done to
            # test the implementation
            try:
                colx = self.corners[(s[mi[idx]] < c) & (c < s[mi[idx + 1]])]
                if len(colx) > 1:  # again, minimum two corners in a row
                    cols.append(colx)
                idx += 1
            except IndexError:
                # print(f"KDE failed at idx = {idx}. Image = {self.img_name}")
                # NOTE: log page id
                break

        self.columns = cols
        return cols

    def correct_corners(self) -> None:
        """
        Objective is to interpolate columns missing one or more corners (rows)
        Approach relies on the most adjacent known point to infer the location
        of the missing

        Returns:
            Inferred corners are added to their respective columns
            in self.columns
        """
        # find mode number of rows across all columns
        # if there is a toss-up, select the height number
        _mode = multimode([len(x) for x in self.columns])
        mode_number_rows = return_largest_value(_mode)
        complete_columns = [i for i in range(len(self.columns)) if len(self.columns[i]) == mode_number_rows]

        self.mode_number_rows = mode_number_rows
        self.complete_columns = complete_columns

        # find problematic columns
        cols_to_interpolate = [i for i in range(len(self.columns)) if i not in complete_columns]

        if not cols_to_interpolate:  # no missing corners
            return

        # find most adjacent complete column to every "problematic" column
        nearest_complete_neighbor = self.list_nearest_from_list(complete_columns, cols_to_interpolate)

        nearest_complete_dict = {k: v for k, v in zip(cols_to_interpolate, nearest_complete_neighbor)}

        for problem, complete in nearest_complete_dict.items():
            prob_col = self.columns[problem]  # the problematic column
            nearest = self.columns[complete]  # nearest complete column
            nearest = nearest[nearest[:, 0].argsort()]  # ensure sort top-bottom
            # get mode x-value for problematic
            mode_x_problematic = mode([x[1] for x in prob_col])
            # make copy of nearest column
            complete_shifted = nearest.copy()
            # shift copy over to problematic
            complete_shifted[:, 1] = mode_x_problematic
            # reshape and ensure sorting of y coordinate
            x_prob = np.sort(prob_col[:, 0].reshape(-1, 1), axis=0)
            x_comp = np.sort(complete_shifted[:, 0].reshape(-1, 1), axis=0)

            A = euclidean_distances(x_comp, x_prob)  # pairwise Eucl. distance
            # find the row(s) that have no overlapping corners
            # -> this is our missing corner
            missing = np.argwhere(np.all(A > self.minD, axis=1))

            for m in missing:
                m = m.flatten()[0]
                nearest_point = nearest[m].flatten()
                nearest_point = nearest_point.reshape(1, 2)
                # get region of interest (ROI) where we should look for corner
                # method used to look at box around ROI, now it considers
                # floor to ceiling og image -> could skip y coordinate
                # ROI = np.hstack([nearest_point[:,0], mode_x_problematic])

                # create fresh mask
                mask = np.zeros((self.HEIGHT, self.WIDTH), dtype="uint8")
                # set box (strip) around ROI
                x1 = max(int(mode_x_problematic - 0.01 * self.WIDTH), 0)
                x2 = min(int(mode_x_problematic + 0.01 * self.WIDTH), self.WIDTH)
                y1 = 0
                y2 = self.HEIGHT
                # only keep ROI in segmentation
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                masked = cv2.bitwise_and(self.seg, self.seg, mask=mask)
                self.draw_on_mask(img=masked, include_horizontal=False)
                # draw horizontal line across image at same height as
                # most adjacent point
                self.skeletonized_mask[int(nearest_point[:, 0])] = 1

                guess = self.run_corner_detector()  # estimated missing corner
                if len(guess) == 0:  # we did not detect a corner candidate
                    # simply return the ROI point as best guess
                    guess = np.hstack([nearest_point[:, 0], mode_x_problematic])

                elif len(guess) > 1:
                    # if we have more than 1 candidate, choose the one closest
                    # to the nearest neighbor
                    idx = np.argmin(np.linalg.norm(guess - nearest_point, axis=1))
                    guess = guess[idx]

                guess = guess.reshape(1, 2)  # ensure shape
                # check that the new corner aligns with nearest
                if not self.check_alignment(guess, nearest_point, also_vertical=False):
                    guess = np.hstack([nearest_point[:, 0], mode_x_problematic])

                # finally, add to column
                self.columns[problem] = np.vstack((self.columns[problem], guess))
                # and to corners
                self.corners = np.vstack((self.corners, guess))
                # NOTE: keep track of interpolated corners -> maybe not
                # self.interpolated_corners.append(guess.flatten())

        # self.interpolated_corners = np.array(self.interpolated_corners)
        return

    def draw_on_mask(self, img: np.ndarray = None, include_horizontal: bool = True) -> None:
        """
        Masks {img} (the segmentation file if nothing else), returns
        skeletonized object
        Args:
            img: raw segmentation unless other is provided
        Returns:
            stores mask as self.skeletonized_mask
        """
        seg = self.seg if img is None else img

        # this part relies on OpenCV for their MORPH_OPEN kernel
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        _, seg = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # specify kernel
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, self.HEIGHT // 4)  # CHANGED: (1, self.HEIGHT//2 ; //3)
        )
        detect_vertical = cv2.morphologyEx(seg, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        # find contours with kernel
        cntsV, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # create mask
        mask = np.zeros((self.HEIGHT, self.WIDTH))
        # draw filled contours on mask
        for c in cntsV:
            mask = cv2.drawContours(mask, [c], -1, (255), thickness=cv2.FILLED)

        if include_horizontal:
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.WIDTH // 5, 1))
            detect_horizontal = cv2.morphologyEx(seg, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
            cntsH, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cntsH:
                mask = cv2.drawContours(mask, [c], -1, (255), thickness=cv2.FILLED)

        # dilate mask to close potential gaps - smooths out a few cases
        dilation_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, dilation_kernel, iterations=3)
        mask = mask // 255  # scale to 0-1
        # mask = skimage.filters.gaussian(mask) # hurts
        footprint = skimage.morphology.disk(3)
        mask = skimage.morphology.erosion(mask, footprint=footprint)  # apply closing
        # skeletonize
        skel = skimage.morphology.skeletonize(mask)
        footprint = skimage.morphology.disk(3)
        skel = skimage.morphology.binary_closing(skel, footprint=footprint)  # apply closing

        self.skeletonized_mask = skel
        '''
        import matplotlib.pyplot as plt
        def show(img):
            plt.imshow(img)
            plt.show()
        import ipdb; ipdb.set_trace()
        '''

    def run_corner_detector(self) -> np.ndarray:
        """
        Returns corners from Harris corner detector on {self.skeletonized_mask}
        """
        # Harris corner detector
        if self.skeletonized_mask is None:
            self.draw_on_mask()

        coords = corner_peaks(
            corner_harris(self.skeletonized_mask),
            min_distance=int(self.minD),
            threshold_rel=0.51,
        )
        corners = corner_subpix(self.skeletonized_mask, coords, window_size=10, alpha=0.51)

        # when the detector is unsure it returns nan (rarely), remove these
        corners = corners[~np.isnan(corners).any(axis=1)]
        return corners

    def find_missing_columns(self) -> None:
        """
        If we have fewer columns than expected, perhaps originating from the
        KDE step, approximate the location of the column(s) by masking
        and KMeans
        Notes:
            This function checks for two things:
            1.  corners that were not found due to an inadequate mask
            2.  missing columns (perhaps due to something with the
                KDE step)
        Returns
            Saves found corners in missing column(s) in new column(s)
        """
        if len(self.columns) == self.IDEAL_NUMBER_OF_COLUMNS:
            return

        median_row_heights = self.median_row_heights()
        self.draw_on_mask(include_horizontal=False)  # only find vertical lines
        # draw horizontal line across at all row heights
        for r in median_row_heights:
            self.skeletonized_mask[int(r)] = 1

        all_corners = self.run_corner_detector()
        A = euclidean_distances(all_corners, self.corners)  # pairwise
        # find the corners that don't overlap
        missing = np.argwhere(np.all(A > self.minD, axis=1))
        if len(missing) > 0:
            # this can happen when a column is missing due to the KDE indexing
            guess = [c.flatten() for c in all_corners[missing]]

            # NOTE: decide what to do about interpolated_corners
            # for now just ignore them: they are only used for inspection
            # if len(self.interpolated_corners) > 0:
            #     self.interpolated_corners = np.vstack((np.array(guess),
            #                                            self.interpolated_corners))
            # else:
            #     self.interpolated_corners = guess
            self.corners = np.vstack((self.corners, np.array(guess)))

        # clear top and bottom
        self.clear_top_and_bottom()

        # get all the corners that are not assigned to a column
        orphans = [c for c in self.corners if not np.any(np.all(c == np.vstack([c for c in self.columns]), axis=1))]
        if not orphans:
            # nothing worked, exiting
            return
        orphans = np.vstack([c for c in orphans])
        # get the number of columns we are missing
        num_missing_columns = self.IDEAL_NUMBER_OF_COLUMNS - len(self.columns)
        # approximate how many columns {orphans} span
        list_inertia_ = []
        for N in range(len(self.columns)):
            col = self.columns[N]
            colX = col[:, 1].reshape(-1, 1)
            kmeans = KMeans(n_clusters=1, random_state=10).fit(colX)
            list_inertia_.append(kmeans.inertia_)

        # NOTE: come up with much better rule:
        ####
        list_inertia_ = np.array(list_inertia_)
        max_inertia_ = np.max(list_inertia_)
        std_inertia_ = np.std(list_inertia_)
        threshold = max_inertia_ + 2 * std_inertia_
        ####

        iner = np.inf
        K = 0
        while iner > 500:  # NOTE: come up with much better rule
            # often inertia_ is in the order of thousand when K is off
            K += 1
            orphansX = orphans[:, 1].reshape(-1, 1)
            kmeans = KMeans(n_clusters=K, random_state=10).fit(orphansX)
            iner = kmeans.inertia_

        for i in range(K):
            if list(kmeans.labels_).count(i) > 1:
                self.columns.append(orphans[np.where(kmeans.labels_ == i)])
            else:
                continue

        # NOTE: move to log -> or maybe just accept
        # assert K >= num_missing_columns, "did not find enough columns"

        return

    def remove_excess_corners(self, median_row_heights: list) -> None:
        """
        Removes excess corner(s) from a column in {self.columns}
        """
        for c in range(len(self.columns)):
            # find out how many is too many
            diff = len(self.columns[c]) - self.mode_number_rows
            if diff > 0:
                colX = self.columns[c][:, 0].reshape(-1, 1)  # only heights
                median_row_heights = np.array(median_row_heights).reshape(-1, 1)
                # pair-wise distance to mode row height
                # assumption: if you are the furthest away, you belong the least
                A = euclidean_distances(colX, median_row_heights)
                # dict of distance to closest row and corresponding index
                dist_dict = {k: v for k, v in enumerate(np.min(A, axis=1))}
                dist_list = sorted(dist_dict.items(), key=lambda x: x[1], reverse=True)
                for i in range(diff):
                    # iteratively delete most distant points
                    idx = dist_list[i][0]
                    self.columns[c] = np.delete(self.columns[c], idx, axis=0)
        return

    def median_row_heights(self) -> list:
        """
        Calculates the mode of the heights for the {self.mode_number_rows}
        """
        median_row_heights = []
        for r in range(self.mode_number_rows):
            height = []
            for col in self.columns:
                col = col[col[:, 0].argsort()]
                height.append(col[r][0])
            median_row_heights.append(int(median(height)))
        return median_row_heights

    def get_idx_blank(self, _list: list, margin: int) -> list:
        """
        Finds index of blank cell(s) (if any) based on [b/w ratio]
        Values are chosen empirically
        Args
            _list: list of cropped out cells
        Returns
            indices to be deleted
        """
        T = 50  # NOTE: chosen empirically
        ratio = []
        for i in _list:
            imgC = i
            # remove 2*margin that was added
            imgC = imgC[
                   # 2*margin:-2*margin,
                   2 * margin: -int(4 * margin + 0.03 * self.HEIGHT),  # CHANGED
                   2 * margin: -2 * margin,
                   ]
            # identify if the images are blank
            img = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 999, T)
            img = 255 - img
            img = img // 255
            img[np.where(np.all(img == 1, axis=1))] = 0  # remove horizontal

            ratio.append(np.sum(img.flatten()) / img.shape[1])
        return [i for i, x in enumerate(ratio) if x < 1]

    def crop_cells(self, column_pair: tuple, margin: float = 0.005):
        """
        Crops out birthdate cells with a margin of {margin}
        Args:
            column_pair: indices for what column to crop. Expects (left,right)
            margin: number of pixels to add around cell during cropping
        Returns:
            saves crops of birthdates to self.birtdate_dir
        """
        margin = int(self.HEIGHT * margin)
        # find coordinates of corners
        self.draw_on_mask()
        # detect corners
        corners = self.run_corner_detector()
        self.corners = corners
        # get columns from corners
        self.get_columns()
        # correct (potential) missing corners / rows
        self.correct_corners()  # should be before find_missing_columns()
        self.find_missing_columns()
        self.correct_corners()  # also needs to be after find_missing_columns()
        # ensure that the order of columns is sorted left to right
        self.columns = sorted(self.columns, key=lambda x: np.mean(x[:, 1]))

        # find mode of row heights
        median_row_heights = self.median_row_heights()
        self.remove_excess_corners(median_row_heights=median_row_heights)

        left, right = column_pair

        col0 = self.columns[left]
        col1 = self.columns[right]
        # sort top to bottom
        col0 = col0[col0[:, 0].argsort()]
        col1 = col1[col1[:, 0].argsort()]
        # ensure correct type
        col0 = col0.astype(int)
        col1 = col1.astype(int)

        # for the columns given by {column_pair}, if they are not reasonably
        # aligned, correct by {median_row_heights}
        _r = 0
        while not self.check_alignment(col0, col1):
            row0 = col0[_r].reshape(1, 2)
            row1 = col1[_r].reshape(1, 2)
            if not self.check_alignment(row0, row1):
                col0[_r, 0] = median_row_heights[_r]
                col1[_r, 0] = median_row_heights[_r]
            _r += 1

        cropped_list = []
        cropped_id_list = []
        coordinates_dict = {}
        # start slicing
        for i in range(len(col0)):
            if i == len(col0) - 1:  # identify last row and cut black edges off
                # look from the bottom for first white pixel mid image
                idx = self.find_first_white_pixel(direction="bottom")
                dlt = int(idx)  # CHANGED: +0.03*self.HEIGHT) # slice off bottom
                # coordinates of cell
                c1 = col0[i, 0] - margin
                c2 = self.HEIGHT - dlt
                c3 = col0[i, 1] - margin
                c4 = col1[i, 1] + margin
                # also get row id
                cc3 = 0
                cc4 = cc4 = col0[i, 1] + margin
                # slice
                cropped = self.img[c1:c2, c3:c4]
                coordinates_dict[i] = [c1, c2, c3, c4]
                cropped_id = self.img[c1:c2, cc3:cc4]

            else:
                c1 = col0[i, 0] - margin
                c2 = col0[i + 1, 0] + margin
                c3 = col0[i, 1] - margin
                c4 = col1[i, 1] + margin
                # row id
                cc3 = 0
                cc4 = col0[i, 1] + margin
                # slice
                cropped_id = self.img[c1:c2, cc3:cc4]
                cropped = self.img[c1:c2, c3:c4]
                coordinates_dict[i] = [c1, c2, c3, c4]
            cropped_list.append(cropped)
            cropped_id_list.append(cropped_id)

        idx2del = self.get_idx_blank(cropped_list, margin=margin)
        cropped_list = [x for i, x in enumerate(cropped_list) if i not in idx2del]
        cropped_id_list = [x for i, x in enumerate(cropped_id_list) if i not in idx2del]
        coordinates_dict = {k: v for k, v in coordinates_dict.items() if k not in idx2del}

        # lastly, update self.corners to only those assigned to a column
        self.corners = [c for c in self.corners if np.any(np.all(c == np.vstack([c for c in self.columns]), axis=1))]

        return cropped_list, cropped_id_list, coordinates_dict
