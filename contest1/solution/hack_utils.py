import os
import tqdm
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from itertools import islice

np.random.seed(1234)
torch.manual_seed(1234)

TRAIN_SIZE = 0.8
NUM_PTS = 971
CROP_SIZE = 128
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"

FLIPPED_IDXS = (
    [i for i in range(64, 128)] +
    [i for i in range(0, 64)] +
    [i for i in range(272, 127, -1)] +
    [i for i in range(337, 401)] +
    [i for i in range(273, 337)] +
    [i for i in range(464, 527)] +
    [i for i in range(401, 464)] +
    [i for i in range(527, 587)] +
    [i for i in range(714, 841)] +
    [i for i in range(587, 714)] +
    [i for i in range(872, 840, -1)] +
    [i for i in range(904, 872, -1)] +
    [i for i in range(936, 904, -1)] +
    [i for i in range(968, 936, -1)] +
    [i for i in range(970, 971)] +
    [i for i in range(969, 970)]
)
FLIPPED_IDXS = torch.as_tensor(FLIPPED_IDXS)


def is_in_jupyter():
    try:
        return 'zmqshell' in str(get_ipython())
    except:
        return False


def stqdm(collection, **kwargs):
    if is_in_jupyter():
        collection = tqdm.tqdm_notebook(collection, **kwargs)
    else:
        collection = tqdm.tqdm(collection, **kwargs)
    return collection


class ScaleMinSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name='image'):
        self.size = torch.tensor(size, dtype=torch.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size=128, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5, elem_name='image'):
        self.p = p
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        sample["img_width"] = w

        if not np.random.binomial(1, self.p):
            sample["flip"] = False
            return sample

        sample["flip"] = True
        sample[self.elem_name] = img[:, ::-1].copy()

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks = landmarks[FLIPPED_IDXS]
            landmarks[:, 0] = torch.tensor((w,), dtype=landmarks.dtype)[:, None] - landmarks[:, 0]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train", train_size=TRAIN_SIZE):
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split is not "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")
        has_landmarks = split in ("train", "val")

        self.image_names = []

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for line in fp)
        num_lines -= 1  # header
        print(f'Total lines (without header): {num_lines}')

        with open(landmark_file_name, "rt") as fp:
            if split == "train":
                length = int(train_size * num_lines)
                print(f'Loading train dataset ({length} lines)')
                lines = islice(fp, 1, length + 1)
            elif split == "val":
                length = num_lines - int(train_size * num_lines)
                print(f'Loading val dataset ({length} lines)')
                lines = islice(fp, int(train_size * num_lines) + 1, None)
            else:
                length = num_lines
                lines = islice(fp, 1, None)

            if has_landmarks:
                self.landmarks = np.empty((length, NUM_PTS, 2), dtype=np.int16)

            for i, line in stqdm(enumerate(lines), total=length, leave=True):
                elements = line.strip().split("\t")
                image_name = os.path.join(images_root, elements[0])
                self.image_names.append(image_name)

                if has_landmarks:
                    landmarks = np.array(elements[1:], dtype=np.int16).reshape((-1, 2))
                    self.landmarks[i] = landmarks

            if has_landmarks:
                self.landmarks = torch.as_tensor(self.landmarks)
            else:
                self.landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]
            sample["landmarks"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image
        sample["image_name"] = self.image_names[idx]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')
