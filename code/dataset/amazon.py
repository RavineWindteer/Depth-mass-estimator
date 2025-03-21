import os
import cv2
import numpy as np

import pickle
import lz4
import lz4.block
from PIL import Image
from io import BytesIO

from dataset.base_dataset import BaseDataset


class amazon(BaseDataset):
    def __init__(self, data_path, filenames_path='./code/dataset/filenames/',
                is_train=True, colored_pc=False, pc_dims=1024, pc_out_dims = 1024):
        super().__init__(colored_pc, pc_dims, pc_out_dims)

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'amazon/')

        txt_path = os.path.join(filenames_path, 'amazon')

        if is_train:
            txt_path += '/file_paths_train.txt'
            self.filenames_list = self.readTXT(txt_path)
        else:
            files_path = './datasets/amazon/test/images/amazon_test_set.pklz'
            self.files = self.get_compressed_object(files_path)

        length = len(self.filenames_list) if is_train else len(self.files)

        phase = 'train' if is_train else 'test'
        print("Dataset: Amazon")
        print("# of %s images: %d" % (phase, length))

    def __len__(self):
        if self.is_train:
            return len(self.filenames_list)
        else:
            return len(self.files)

    def __getitem__(self, idx):
        if self.is_train:
            pklz_path = self.data_path + self.filenames_list[idx]
            pklz_path = pklz_path.replace('/train/', '/train/images/')
            image, mass, volume, density = self.get_amazon_test_set(pklz_path)
            path_depth = pklz_path.replace('/images/', '/depth/').replace('.pklz', '.png')
            filename = path_depth.split('/')[-1]
        else:
            image, mass, volume, density = self.get_amazon_data_test(idx)
            path_depth = './datasets/amazon/test/depth/' + str(idx) + '.png'
            filename = str(idx) + '_amazon_test'

        depth = cv2.imread(path_depth, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.is_train:
            image, pc_input, _ = self.augment_training_data(image, depth, None)
        else:
            pc_input = self.get_pointcloud(depth)
        
        pc_input = self.to_tensor(pc_input)
        image = self.convert_to_densenet_input(image)

        return {'image': image, 'pc_incomplete': pc_input, 'pc_sparse': False, 'pc_complete': False, 'mass': mass, 'volume': volume, 'density': density, 'filename': filename}
    
    def get_compressed_object(self, filename):
        with open(filename, 'rb') as fp:
            compressed_bytes = fp.read()
        decompressed = lz4.block.decompress(compressed_bytes)
        pickled_object = pickle.loads(decompressed)

        return pickled_object
    
    def unpack_amazon_image(self, binary_image_data):
        bytes_image_data = BytesIO(binary_image_data)
        image = Image.open(bytes_image_data)
        opencv_image = np.array(image)

        return opencv_image
    
    def resize_image(self, image):
        border_size = (640 - 480) // 2
        border_color = [255, 255, 255]
        image = cv2.resize(image, (480, 480))
        image = cv2.copyMakeBorder(image, 0, 0, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

        return image
    
    def get_amazon_test_set(self, path):
        data = self.get_compressed_object(path)
        mass = float(data['weight'])
        binary_image_data = data['image_data']

        mass = mass * 0.45359237 # Convert from lbs to kg

        # Get aligned dims
        dimensions = data['dimensions']
        dimensions = np.array([float(dim) for dim in dimensions])
        dimensions = dimensions * 2.54 # Convert from inches to cm
        dimensions = dimensions / 100.0 # Convert from cm to meters
        volume = (dimensions[0] + 0.005) * (dimensions[1] + 0.005) * (dimensions[2] + 0.005)

        density = mass / volume

        image = self.unpack_amazon_image(binary_image_data)
        image = self.resize_image(image)

        return image, mass, volume, density
    
    def get_amazon_data_test(self, idx):
        img, mask, density, dims, rect, weight = self.files[idx]

        mass = weight * 0.45359237 # Convert from lbs to kg

        # Get aligned dims
        dimensions = np.array([float(dim) for dim in dims])
        dimensions = dimensions * 2.54 # Convert from inches to cm
        dimensions = dimensions / 100.0 # Convert from cm to meters
        volume = (dimensions[0] + 0.005) * (dimensions[1] + 0.005) * (dimensions[2] + 0.005)

        density = mass / volume

        image = np.array(img)
        image = self.resize_image(image)

        return image, mass, volume, density
    