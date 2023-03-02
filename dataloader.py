import glob
import os

import albumentations as A
import cv2
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir=str, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.data = []
        self.class_map = {}
        self.extensions = ("jpeg", "jpg", "png")

        file_list = sorted(glob.glob(self.data_dir + "/*"))

        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*"):
                ext = img_path.split("/")[-1].split(".")[-1]
                if ext in self.extensions:
                    self.data.append([img_path, class_name])

        for idx, class_path in enumerate(file_list):
            class_name = class_path.split("/")[-1]
            self.class_map[class_name] = idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        # Applying transforms on image
        if self.transforms:
            img = self.transforms(image=img)["image"]

        label = self.class_map[class_name]

        return img, label


def get_transform():
    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return A.Compose([resize, normalize])


if __name__ == "__main__":
    data_dir = os.path.join(os.path.abspath(__file__ + "/../../"), "data/images/")
    dataset = CustomDataset(data_dir=data_dir, transforms=get_transform())
    # print(len(dataset))
    # print(dataset[0][0].shape)

    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    total_imgs = 0
    for imgs, labels in data_loader:
        total_imgs += int(imgs.shape[0])
        break
    print(total_imgs)
