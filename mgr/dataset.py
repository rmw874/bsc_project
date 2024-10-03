from pathlib import Path

import numpy as np
import torch
from PIL import Image
from albumentations import Compose, DualTransform, Resize
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import measure, morphology
from skimage.draw import polygon
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
from tqdm import tqdm


class MGRDataset(Dataset):
    def __init__(self, folders, transform=None, img_augment=None, in_memory=False,
                 memory_file=None, return_name=False, img_size=(1024 + 128, 1024)):
        self.folders = folders
        self.transform = transform
        self.img_augment = img_augment
        self.in_memory = in_memory
        self.return_name = return_name
        self.img_size = img_size

        self.data = []
        for folder in self.folders:
            folder = Path(folder)
            if not folder.is_dir():
                raise ValueError(f"'{folder}' is not a folder.")
            img_files = folder.glob('img.*')
            img_files = list(img_files)
            if len(img_files) != 1:
                raise ValueError(f"Folder '{folder}' should contain exactly one image file.")

            img_path = img_files[0]
            label_files = list(folder.glob("label.*"))
            if len(label_files) != 1:
                raise ValueError(
                    f"Folder '{folder}' should contain exactly one label file.")

            label_path = label_files[0]
            self.data.append((img_path, label_path))

        self.memory = []
        if in_memory:
            if memory_file is not None and Path(memory_file).is_file():
                # memory file overwrites self.data
                memory = torch.load(memory_file)
                self.data = memory["data"]
                self.memory = memory["memory"]
            else:
                self.memory = [self.get_sample(i) for i in tqdm(range(len(self)), desc="Data preloading")]
                if memory_file is not None:
                    torch.save({"data": self.data, "memory": self.memory}, memory_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.in_memory:
            image, label = self.memory[idx]
        else:
            image, label = self.get_sample(idx)

        image, label = image.astype(np.single), label.astype(int)
        image = rescale_intensity(image, out_range=(0, 1))

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        if self.return_name:
            path = self.data[idx][0]
            return image, label, f"{path.parent.name}_{path.stem}"
        else:
            return image, label

    def get_sample(self, idx):
        img_path, label_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        bbox = image.getbbox()  # remove obvious black outside area
        image = image.crop(bbox)
        label = Image.open(label_path).convert('RGB')
        label = label.crop(bbox)

        if self.img_size is not None:
            image = image.resize(self.img_size)
            label = label.resize(self.img_size)

        image = np.array(image)
        label = np.array(label, dtype=np.single)
        # we know that "red" is the segmentation we are looking for
        # filter and threshold at 10 to allow for cleaner annotations
        label = (gaussian_filter(label[..., [0]], sigma=1) > 10).astype(image.dtype)

        return image, label.astype(bool)


class MGRDatasetColumns(MGRDataset):
    def __init__(self, folders, column_ids: [tuple, list], transform=None, img_augment=None, in_memory=False,
                 memory_file=None, return_name=False, img_size=(1024 + 128, 1024), column_count: int = 9,
                 check_row_counts: bool = True):
        assert len(column_ids) >= 1, f"have to provide at least one column to extract"
        self.out_points_thresh = 50
        self.out_points_step = 5
        self.column_radius_thresh = 30
        self.cell_area_thresh = 1230
        self.cell_margin = 20
        self.check_row_counts = check_row_counts
        self.column_count = column_count
        self.column_ids = column_ids
        super().__init__(folders, transform, img_augment, in_memory, memory_file, return_name, img_size)

    def get_sample(self, idx):
        image, label = super().get_sample(idx)

        l = label.astype(int).squeeze(-1)
        # get a concave hull by selecting outer point and then drawing polygon

        try:
            out_points = find_outer_points(l, self.out_points_thresh, self.out_points_step)
        except AssertionError as e:
            raise Exception(f"Files: {self.data[idx]} gave: {str(e)}")

        op = np.concatenate(out_points, 0)

        sk = np.zeros_like(l)
        rr, cc = polygon(op[:, 0], op[:, 1])
        # Fill the space between the outer points
        sk[rr, cc] = 1

        # get cells of table by doing connected component analyses on the concave hull without the label lines
        cells = l + (1 - sk)
        cells = morphology.dilation(morphology.erosion(measure.label(cells, background=1)))

        for c in np.unique(cells):
            if c == 0: continue  # skip background
            cell = c == cells
            if cell.sum() < self.cell_area_thresh:  # skip if too small
                cells[cell] = 0
                continue

        # enlarge cells until they touch (first horizontal then vertically)
        new_cells = cells.copy()
        n_unique = len(np.unique(cells))
        while n_unique == len(np.unique(new_cells)):
            cells = new_cells.copy()
            new_cells = measure.label(morphology.dilation(new_cells > 0, np.ones((1, 3))))

        new_cells = cells.copy()
        n_unique = len(np.unique(cells))
        while n_unique == len(np.unique(new_cells)):
            cells = new_cells.copy()
            new_cells = measure.label(morphology.dilation(new_cells > 0, np.ones((3, 1))))

        # do not increase outside the outer points
        cells[(1 - sk) > 0] = 0

        try:
            columns = get_columns_rows(
                cells, self.column_ids, self.column_radius_thresh, self.column_count,
                self.check_row_counts, self.cell_margin
            )
        except AssertionError as e:
            raise Exception(f"Files: {self.data[idx]} gave: {str(e)}")

        return image, columns  # .astype(bool)

    def __getitem__(self, idx):
        if self.in_memory:
            image, label = self.memory[idx]
        else:
            image, label = self.get_sample(idx)

        image, label = image.astype(np.single), label.astype(int)
        image = rescale_intensity(image, out_range=(0, 1))
        n_rows = np.amax(label)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        if self.return_name:
            path = self.data[idx][0]
            return image, label, n_rows, f"{path.parent.name}_{path.stem}"
        else:
            return image, label, n_rows


class MGRDatasetInference(Dataset):
    def __init__(self, folders, transform=None, img_augment=None, img_size=(1024 + 128, 1024)):
        self.folders = folders
        self.transform = transform
        self.img_augment = img_augment
        self.img_size = img_size

        self.data = []
        for folder in self.folders:
            folder = Path(folder)
            if folder.is_dir():
                img_files = folder.glob('*')
                img_files = [f for f in img_files if
                             f.suffix.lower() in [".png", ".jpg", ".tiff", ".tif", ".jpeg"]]
                if len(img_files) == 0:
                    raise ValueError(f"Folder '{folder}' should contain at least one image.")
                self.data.extend(img_files)
            else:
                assert folder.suffix.lower() in [".png", ".jpg", ".tiff", ".tif", ".jpeg"], f"{folder} is not an image"
                self.data.append(folder)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        bbox = image.getbbox()  # remove obvious black outside area
        image = image.crop(bbox)
        img_ori = np.array(image)
        if self.img_size is not None:
            image = image.resize(self.img_size)
        image = np.array(image, dtype=np.single)
        image = rescale_intensity(image, out_range=(0, 1))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, f"{img_path.parent.name}_{img_path.stem}", img_ori


def get_columns_rows(cells, column_ids: [list, tuple], threshold_radius: int,
                     column_count: int = 9, check_row_counts: bool = True, cell_margin: int = 20):
    # get each cells center position and identify the highest cell
    x_coords = np.arange(cells.shape[0])  # Array of x coordinates
    y_coords = np.arange(cells.shape[1])  # Array of y coordinates
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    top_cell_h = cells.shape[1]
    cell_hs = []
    cell_ws = []
    cell_cs = []
    for c in np.unique(cells):
        if c == 0: continue  # skip background
        cell = c == cells

        cell_h = X[cell].mean()
        cell_hs.append(cell_h)
        cell_ws.append(Y[cell].mean())
        cell_cs.append(c)
        if cell_h < top_cell_h:
            top_cell_h = cell_h

    cell_hs = np.array(cell_hs)
    cell_ws = np.array(cell_ws)
    cell_cs = np.array(cell_cs)

    # get header
    # get cells at similar height of top cell
    header_m = (cell_hs <= top_cell_h + threshold_radius) & (cell_hs >= top_cell_h - threshold_radius)
    header_ws = cell_ws[header_m]
    header_cs = cell_cs[header_m]

    # sort by width to get order
    header_idx = np.argsort(header_ws)
    header_ws = header_ws[header_idx]
    header_cs = header_cs[header_idx]

    # write header
    header = np.zeros_like(cells)
    for i, c in enumerate(header_cs):
        header[cells == c] = i + 1

    # assert if all columns are found
    assert header.max() == column_count, f"found {header.max()} columns instead of {column_count}"

    # extract relevant columns and rows based on headers
    columns = []
    column_min_h = []
    column_max_h = []

    for c_id in column_ids:
        column = np.zeros_like(cells)
        head_w = header_ws[c_id - 1]

        # get cells at same width
        column_m = (cell_ws <= head_w + threshold_radius) & (cell_ws >= head_w - threshold_radius)
        column_c = cell_cs[column_m]

        # write to columns array
        c_min_h = []
        c_max_h = []
        for c_i, c in enumerate(column_c):
            cell = c == cells
            column[cell] = c_i + 1
            cell_height_bounds = np.flatnonzero(cell.any(1))
            c_min_h.append(cell_height_bounds[0])
            c_max_h.append(cell_height_bounds[-1])

        columns.append(column)
        column_min_h.append(c_min_h)
        column_max_h.append(c_max_h)

    columns = np.stack(columns, -1)

    if check_row_counts:
        column_counts = columns.max((0, 1))
        first_len = column_counts[0]
        inconsistency_msg = f"inconsistency detected:"
        for i, c_idx in enumerate(column_ids):
            inconsistency_msg += f"{c_idx}: {column_counts[i]}\n"

        assert first_len >= 2 and all([first_len == count for count in column_counts]), inconsistency_msg

        column_min_h = np.array(column_min_h)
        column_max_h = np.array(column_max_h)
        for i in range(first_len):
            # get height bounds (do that now before resizing for consistency across images)

            std_h_min = np.std([h for h in column_min_h[:, i]])
            std_h_max = np.std([h for h in column_max_h[:, i]])

            assert (std_h_max <= cell_margin and i != first_len - 1) or i == first_len - 1, \
                f"inconsistency detected: row {i + 1} has large top height standard deviation: {std_h_max:.3f}"
            assert std_h_min <= cell_margin, f"inconsistency detected: row {i + 1} " \
                                             f"has large bottom height standard deviation: {std_h_min:.3f}"

    return columns


def cluster_lines(Z, Q, X, Y, thresh_radius, step_size, op: str):
    if op == "max":
        min_x = np.max(Z)
    else:
        min_x = np.min(Z)

    # decrease filter until enough separate clusters emerge
    i = 0
    while True:
        t_radius = thresh_radius - i * step_size

        if op == "max":
            radius = (Z >= min_x - t_radius)
        else:
            radius = (Z <= min_x + t_radius)
        candidate = np.array([[x, y] for x, y in zip(X[radius], Y[radius])])
        q = Q[radius][:, None]

        dbscan = DBSCAN(t_radius, min_samples=1)
        cluster = dbscan.fit_predict(q)
        i += 1
        if len(np.unique(cluster)) > 1:
            # ensure no clustering of multiple lines
            std = [q[cluster == c].std() for c in np.unique(cluster)]
            if max(std) * 3 < t_radius:
                return cluster, candidate
        assert thresh_radius - i * step_size > 0, "couldn't find more than one point"


def find_outer_points(seg, thresh_radius=20, step_size=10):
    x_coords = np.arange(seg.shape[0])  # Array of x coordinates
    y_coords = np.arange(seg.shape[1])  # Array of y coordinates
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    mask = seg > 0
    X = X[mask]
    Y = Y[mask]

    # bottom
    cluster, candidate = cluster_lines(X, Y, X, Y, thresh_radius, step_size, "max")
    pts = []
    for c in np.unique(cluster):
        cand = candidate[cluster == c]
        idx = np.argmax(cand[:, 0])
        pts.append(cand[idx])
    bottom = np.array(pts)

    # left    
    cluster, candidate = cluster_lines(Y, X, X, Y, thresh_radius, step_size, "min")
    pts = []
    for c in np.unique(cluster):
        cand = candidate[cluster == c]
        idx = np.argmin(cand[:, 1])
        pts.append(cand[idx])
    left = np.array(pts)
    argsort = np.argsort(left[:, 0])
    left = left[argsort][::-1]

    # top
    min_x = np.min(X)
    radius = (X <= min_x + thresh_radius)
    candidate = np.array([[x, y] for x, y in zip(X[radius], Y[radius])])
    # random for top since we have the top line
    argsort = np.argsort(candidate[:, 1])
    candidate = candidate[argsort]
    cluster = np.linspace(0, 8, len(argsort), endpoint=False)
    cluster[cluster % 1 > 0.9] = -1
    cluster = cluster.astype(int)
    pts = []
    for c in np.unique(cluster):
        if c == -1:
            pass
        cand = candidate[cluster == c]
        idx = np.argmin(cand[:, 0])
        pts.append(cand[idx])
    top = np.array(pts)
    argsort = np.argsort(top[:, 1])
    top = top[argsort]

    # right
    cluster, candidate = cluster_lines(Y, X, X, Y, thresh_radius, step_size, "max")
    pts = []
    for c in np.unique(cluster):
        cand = candidate[cluster == c]
        idx = np.argmax(cand[:, 1])
        pts.append(cand[idx])
    right = np.array(pts)
    argsort = np.argsort(right[:, 0])
    right = right[argsort]

    # this adds an extra point in the lower left and right corner
    left_bottom = np.argmax(left[:, 0])
    left_bottom = left[left_bottom][1]

    bottom_left = np.argmin(bottom[:, 1])
    bottom_left = bottom[bottom_left][0]

    right_bottom = np.argmin(right[:, 0])
    right_bottom = right[right_bottom][1]

    bottom_right = np.argmax(bottom[:, 1])
    bottom_right = bottom[bottom_right][0]

    bottom = np.concatenate([[[bottom_right, right_bottom]], bottom, [[bottom_left, left_bottom]]], 0)
    argsort = np.argsort(bottom[:, 1])
    bottom = bottom[argsort][::-1]

    return left, top, right, bottom


class WatershedFromLabels(DualTransform):
    def __init__(self, num_energy_levels: int, down_scale_factor: int = 1):
        super().__init__(always_apply=True)
        self.num_energy_levels = num_energy_levels
        self.down_scale_factor = down_scale_factor

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
        }

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        if self.down_scale_factor > 1:
            out_shape = (img.shape[0] // self.down_scale_factor, img.shape[1] // self.down_scale_factor)
            img = resize(img, out_shape, order=0, mode='constant', anti_aliasing=False)

        fe = np.zeros_like(img)
        # We rasterize each object independently to ensure a clean edge.
        for i in range(img.shape[2]):
            m = img[..., i]
            for v in np.unique(m):
                if v == 0:
                    continue
                e = distance_transform_edt(m == v)
                fe[..., i] = np.maximum(fe[..., i], e)

        labels = np.ceil(fe)

        # bring energy levels into required shape
        y_energy = np.clip(labels, a_min=0, a_max=self.num_energy_levels)
        y_energy = torch.nn.functional.one_hot(
            torch.tensor(y_energy).long(), self.num_energy_levels + 1
        )[..., 1:].numpy()
        y_energy = np.flip(np.flip(y_energy, -1).cumsum(-1), -1)
        y_energy = y_energy.reshape((*y_energy.shape[:2], -1))

        return y_energy


if __name__ == '__main__':
    # Example usage:
    from glob import glob
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb

    # Define transformations
    n_energy = 5
    down_scale_factor = 1
    transform = Compose([
        Resize(256, 256),
        WatershedFromLabels(n_energy, down_scale_factor=down_scale_factor),
        ToTensorV2(transpose_mask=True),

    ])

    # List of folders
    folders = [f for f in glob("../../*data202?/*") if Path(f).is_dir()]
    # folders = [f for f in glob("../../*data202?/28") if Path(f).is_dir()]

    # Create dataset instance
    dataset = MGRDatasetColumns(
        folders=folders, column_ids=[1, 2, 4, 5], transform=transform, check_row_counts=True,
    )

    # debug
    # ["" for data in dataset]

    # Example usage
    image, label, n_row = dataset[0]
    label = torch.nn.functional.upsample_nearest(
        label.unsqueeze(0).float(), (image.shape[-2], image.shape[-1])
    ).squeeze(0).long()
    label = label.reshape(4, n_energy, *label.shape[-2:]).sum(1).sum(0).numpy()
    color = label2rgb(label, image.movedim(0, -1).numpy())
    f, ax = plt.subplots(1)
    ax.imshow(color)
    print(image.shape)  # Shape of the image tensor
    print(label.shape)  # Shape of the label tensor
    plt.show()

    # Create dataset instance
    dataset = MGRDataset(folders=folders, transform=transform)

    # Example usage
    image, label = dataset[0]
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(image.movedim(0, -1))
    ax[1].imshow(label.movedim(0, -1))
    print(image.shape)  # Shape of the image tensor
    print(label.shape)  # Shape of the label tensor
    plt.show()
