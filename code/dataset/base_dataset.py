import numpy as np
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.transforms import CustomHorizontalFlip, CustomVerticalFlip
from PIL import Image
import open3d as o3d


def get_dataset(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    dataset_lib = importlib.import_module(
        '.' + dataset_name, package='dataset')
    dataset_abs = getattr(dataset_lib, dataset_name)

    return dataset_abs(**kwargs)


class BaseDataset(Dataset):
    def __init__(self, colored_pc, pc_dims, pc_out_dims):

        self.colored_pc = colored_pc
        self.pc_dims = pc_dims
        self.pc_out_dims = pc_out_dims

        self.count = 0

        self.horizontal_transform = CustomHorizontalFlip()
        self.vertical_transform = CustomVerticalFlip()
        
        basic_transform = [
            self.horizontal_transform,
            self.vertical_transform,
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]

        preprocess_densenet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.basic_transform = basic_transform
        self.preprocess_densenet = preprocess_densenet
        self.to_tensor = transforms.ToTensor()

    def readTXT(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT

    def augment_training_data(self, image, depth, model=None):
        H, W, C = image.shape

        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)
        image = augmented['image']
        depth = augmented['depth']

        pc_gt = None

        if model is not None:
            if self.horizontal_transform.applied:
                #model.scale((-1, 1, 1), center=(0, 0, 0))
                model.vertices = o3d.utility.Vector3dVector(np.asarray(model.vertices) * [-1, 1, 1])
            if self.vertical_transform.applied:
                #model.scale((1, 1, -1), center=(0, 0, 0))
                model.vertices = o3d.utility.Vector3dVector(np.asarray(model.vertices) * [1, 1, -1])

            pc_gt = self.get_pc_gt(model)

        self.horizontal_transform.clear()
        self.vertical_transform.clear()

        if self.colored_pc:
            pointcloud = self.get_pointcloud(depth, image)
        else:
            pointcloud = self.get_pointcloud(depth)

        self.count += 1

        return image, pointcloud, pc_gt
    
    def convert_to_densenet_input(self, image):
        input_tensor = self.preprocess_densenet(Image.fromarray(image))
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch
    
    def get_pointcloud(self, depth, image = None):
        width = 640
        height = 480
        cx = width / 2.0
        cy = height / 2.0
        fx = 688 #588
        fy = 688 #588

        # Create 3D coordinates
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        x, y = np.meshgrid(x, y)

        # Normalize x and y coordinates and scale by depth map
        normalized_x = (x - cx) / fx * depth
        normalized_y = (y - cy) / fy * depth

        # Stack x, y, depth, and colors to create 3D point cloud
        point_cloud = np.dstack((normalized_x, normalized_y, depth)).astype(np.float32)

        if image is not None:
            point_cloud = np.concatenate((point_cloud, image), axis=2)
            point_cloud = point_cloud.reshape(-1, 6) # (number of points, 6)
        else:
            point_cloud = point_cloud.reshape(-1, 3) # (number of points, 3)
        
        # Remove points with depth == 0
        point_cloud = point_cloud[point_cloud[:, 2] != 0]

        # Randomly select pc_dims points
        if point_cloud.shape[0] > self.pc_dims:
            idx = np.random.choice(point_cloud.shape[0], self.pc_dims, replace=False)
            point_cloud = point_cloud[idx]

            # Center the point cloud
            center = np.mean(point_cloud, axis=0)
            point_cloud -= center
            point_cloud = point_cloud / 5.0
        else:
            # Center the point cloud
            center = np.mean(point_cloud, axis=0)
            point_cloud -= center
            point_cloud = point_cloud / 5.0
            
            point_cloud = np.pad(point_cloud, ((0, self.pc_dims - point_cloud.shape[0]), (0, 0)), 'constant')
        
        return point_cloud
    
    def get_pc_gt(self, model):
        pc_gt = model.sample_points_uniformly(number_of_points=self.pc_out_dims)
        pc_gt = np.asarray(pc_gt.points)

        return pc_gt
