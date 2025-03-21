'''
Ricardo Cardoso, 2024
'''

import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging_ as logging
from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.inference_options import InferenceOptions

import matplotlib.pyplot as plt


def cropping_img(args, pred, input_RGB):
    pred[torch.isinf(pred)] = args.max_depth_eval
    pred[torch.isnan(pred)] = args.min_depth_eval

    input_RGB = input_RGB.squeeze(0).numpy()

    # https://www.geeksforgeeks.org/image-segmentation-with-watershed-algorithm-opencv-python/
    gray = cv2.cvtColor(input_RGB, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY_INV) # cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

    valid_mask = torch.from_numpy(bin_img > (255 / 2)).to(device=pred.device)

    '''
    background_color_tensor = torch.tensor(args.background_color, dtype=input_RGB.dtype, device=input_RGB.device)
    background_color_tensor = background_color_tensor[None, :, None, None]
    background_color_tensor = background_color_tensor.expand_as(input_RGB)
    mask_r = (input_RGB[0,0,:,:] != background_color_tensor[0,0,:,:])
    mask_g = (input_RGB[0,1,:,:] != background_color_tensor[0,1,:,:])
    mask_b = (input_RGB[0,2,:,:] != background_color_tensor[0,2,:,:])
    valid_mask = mask_r & mask_g & mask_b
    valid_mask = valid_mask.squeeze(0)
    '''

    eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
    eval_mask[45:471, 41:601] = 1
    valid_mask = torch.logical_and(valid_mask, eval_mask)

    zeros = torch.zeros_like(pred)
    valid_mask_squeezed = valid_mask.squeeze(0)
    new_pred = torch.where(valid_mask_squeezed, pred, zeros)

    return new_pred


def main():
    # experiments setting
    opt = InferenceOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    result_path = os.path.join(args.result_dir, args.exp_name)
    logging.check_and_make_dirs(result_path)
    print("Saving result images in to %s" % result_path)

    print("\n1. Define Model")
    model = GLPDepth(max_depth=args.max_depth, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n2. Define Dataloader")
    dataset_kwargs = {'data_path': args.data_path, 'dataset_name': args.dataset,
                          'is_train': False}

    test_dataset = get_dataset(**dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True)

    print("\n3. Inference & Evaluate")
    for batch_idx, batch in enumerate(test_loader):
        input_RGB = batch['image'].to(device) # shape = (1, 3, 480, 640)
        filename = batch['filename']
        normalization = batch['normalization'].to(device)
        input_RGB_no_tensor = batch['image_no_tensor']

        with torch.no_grad():
            pred = model(input_RGB)
        pred_d = pred['pred_d'].squeeze() * normalization[0]
        
        save_path = os.path.join(result_path, filename[0])
        if save_path.split('.')[-1] == 'jpg':
            save_path = save_path.replace('jpg', 'png')
        
        pred_d = pred_d * 1000.0
        pred_d = cropping_img(args, pred_d, input_RGB_no_tensor)
        pred_d = pred_d.cpu().numpy()

        cv2.imwrite(save_path, pred_d.astype(np.uint16),
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        '''
        save_path = os.path.join(result_path, filename[0])
        pred_d_numpy = pred_d.squeeze() #pred_d.squeeze().cpu().numpy()
        pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        cv2.imwrite(save_path, pred_d_color)
        '''
        
        logging.progress_bar(batch_idx, len(test_loader), 1, 1)
            
    print("Done")


if __name__ == "__main__":
    main()