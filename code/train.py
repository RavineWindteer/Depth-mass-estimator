import os
import cv2
import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.writer import DataWriter
from torch.nn.utils import clip_grad_norm_

import utils.logging_ as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
from models.depth2mass import Depth2Mass
from utils.chamfer_dist import chamfer_distance_bidirectional
from utils.alternate_loader import alternate_loader
from utils.metrics import compute_metrics_mass


metric_name = ['alde', 'ape', 'mnre', 'rmse_log', 'log10', 'silog', 'q']


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    # Logging
    exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
    logging.check_and_make_dirs(log_dir)
    writer = DataWriter(log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')
    logging.log_args_to_txt(log_txt, args)

    # Create result directory
    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Load model
    model = Depth2Mass(args.emb_dims, args.pc_in_dims, args.pc_model, args.pc_completion, args.pc_out_dims)

    # Load checkpoint if provided
    if args.ckpt_dir is not None:
        model_weight = torch.load(args.ckpt_dir)
        if 'module' in next(iter(model_weight.items()))[0]:
            model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
        model.load_state_dict(model_weight)
        print('Loaded model from %s' % args.ckpt_dir)

    # CPU-GPU agnostic settings
    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model.to(device)

    # Load dataset 1
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)
    train_loader1 = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_1,
                                               shuffle=True, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True)
    val_loader1 = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True)
    
    # Load dataset 2
    train_loader2 = None
    val_loader2 = None
    if args.use_2_datasets:
        dataset_kwargs = {'dataset_name': args.dataset2, 'data_path': args.data_path}
        train_dataset = get_dataset(**dataset_kwargs)
        val_dataset = get_dataset(**dataset_kwargs, is_train=False)
        train_loader2 = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_2,
                                                shuffle=True, num_workers=args.workers, 
                                                pin_memory=True, drop_last=True)
        val_loader2 = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                pin_memory=True)
    
    # Create the alternate loader
    if train_loader2 is not None:
        train_loader = alternate_loader(train_loader1, train_loader2)
        val_loader = alternate_loader(val_loader1, val_loader2)
    else:
        train_loader = train_loader1
        val_loader = val_loader1
    
    global train_loader_len
    if train_loader2 is not None:
        train_loader_len = len(train_loader1) + len(train_loader2)
    else:
        train_loader_len = len(train_loader1)

    global val_loader_len
    if val_loader2 is not None:
        val_loader_len = len(val_loader1) + len(val_loader2)
    else:
        val_loader_len = len(val_loader1)

    # Training settings
    optimizer = optim.Adam(model.parameters(), args.lr)

    global global_step
    if train_loader2 is not None:
        global_step = (len(train_loader1) + len(train_loader2)) * (args.start_epoch - 1)
    else:
        global_step = len(train_loader1) * (args.start_epoch - 1)

    # Perform experiment
    for epoch in range(args.start_epoch, args.epochs + 1):
        # Create the alternate loader
        if train_loader2 is not None:
            train_loader = alternate_loader(train_loader1, train_loader2)
            val_loader = alternate_loader(val_loader1, val_loader2)
        else:
            train_loader = train_loader1
            val_loader = val_loader1
        
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, model, optimizer=optimizer,
                           device=device, epoch=epoch, args=args)
        writer.add_scalar('Training loss', loss_train)
        
        results_dict, loss_val, pc_loss, mass_loss = validate(val_loader, model, device=device,
                            epoch=epoch, args=args, log_dir=log_dir)
        writer.add_scalar('Val loss', loss_val)
        if pc_loss is not None:
            writer.add_scalar('PC loss', pc_loss)
            writer.add_scalar('Mass loss', mass_loss)

        result_lines = logging.display_result(results_dict)
        print(result_lines)

        with open(log_txt, 'a') as txtfile:
            txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
            txtfile.write(result_lines)                

        for each_metric, each_results in results_dict.items():
            writer.add_scalar(each_metric, each_results)
    
    writer.save()


def train(train_loader, model, optimizer, device, epoch, args):
    torch.cuda.empty_cache()
    global global_step
    model.eval() # model.train()
    global train_loader_len

    average_loss = logging.AverageMeter()
    average_loss_pc = logging.AverageMeter()
    average_loss_mass = logging.AverageMeter()
    half_epoch = args.epochs // 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4
    
    max_lr = 3.5e-5 #1e-4
    #mid_lr = 6.5e-5
    min_lr = 8e-6

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1
        torch.cuda.empty_cache()
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            if global_step < train_loader_len * half_epoch:
                current_lr = (max_lr - min_lr) * (global_step /
                                              train_loader_len/half_epoch) ** 0.9 + min_lr
            else:
                current_lr = (min_lr - max_lr) * (global_step /
                                              train_loader_len/half_epoch - 1) ** 0.9 + max_lr
            param_group['lr'] = current_lr
        
        # Get input data
        image = batch['image'].to(device)
        pc_incomplete = batch['pc_incomplete'].to(device)
        mass = batch['mass'].unsqueeze(1).to(device)
        pc_gt = batch['pc_complete'].to(device)
        is_pc_gt = (len(batch['pc_complete'][0].size()) != 0)

        # Forward pass
        preds = model((pc_incomplete, image))

        # Get predictions
        pc_reconstructed = None
        if args.pc_completion:
            mass_pred, pc_reconstructed = preds
        else:
            mass_pred = preds
        
        # Compute loss
        optimizer.zero_grad()

        loss_mass = None
        loss_pc = None
        if not is_pc_gt:
            # Compute mass loss (ALDE loss -> Absolute Log Difference Error)
            loss_mass = torch.mean(torch.abs(torch.log(mass_pred) - torch.log(mass)))
            loss_total = loss_mass
        else:
            pc_incomplete = pc_incomplete.squeeze(dim = 1)
            loss_pc = chamfer_distance_bidirectional(pc_gt, pc_reconstructed)
            loss_total = loss_pc

        
        # Check if loss_total is NaN
        if not torch.isnan(loss_total):
            average_loss.update(loss_total.item(), mass.shape[0])
            if not is_pc_gt:
                average_loss_mass.update(loss_mass.item(), mass.shape[0])
            else:
                average_loss_pc.update(loss_pc.item(), pc_gt.shape[0])
            loss_total.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            if args.pc_completion:
                if is_pc_gt:
                    logging.progress_bar(batch_idx, train_loader_len, args.epochs - args.start_epoch, epoch,
                                    ('Total Loss: %.4f (%.4f); Mass Loss: %.4f (%.4f); PC Loss: %.4f (%.4f)' %
                                    (average_loss.val, average_loss.avg, average_loss_mass.avg,
                                    average_loss_mass.avg, average_loss_pc.val, average_loss_pc.avg)))
                else:
                    logging.progress_bar(batch_idx, train_loader_len, args.epochs - args.start_epoch, epoch,
                                    ('Total Loss: %.4f (%.4f); Mass Loss: %.4f (%.4f); PC Loss: %.4f (%.4f)' %
                                    (average_loss.val, average_loss.avg, average_loss_mass.val,
                                    average_loss_mass.avg, average_loss_pc.avg, average_loss_pc.avg)))
            else:
                logging.progress_bar(batch_idx, train_loader_len, args.epochs - args.start_epoch, epoch,
                                ('Total Loss: %.4f (%.4f); Mass Loss: %.4f (%.4f)' %
                                (average_loss.val, average_loss.avg, average_loss_mass.val, average_loss_mass.avg)))
                
            optimizer.step()
        else:
            logging.progress_bar(batch_idx, train_loader_len, args.epochs - args.start_epoch, epoch,
                            ('Total Loss: %s (%.4f); Mass Loss: %s (%.4f); PC Loss: %s (%.4f)' %
                            ('NaN', average_loss.avg, 'NaN', average_loss_mass.avg, 'NaN', average_loss_pc.avg)))
            

    return loss_total


def validate(val_loader, model, device, epoch, args, log_dir):
    average_loss = logging.AverageMeter()
    average_loss_pc = logging.AverageMeter()
    average_loss_mass = logging.AverageMeter()
    global val_loader_len
    model.eval()

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(
            log_dir, 'epoch_%02d_model.ckpt' % epoch))

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    mass_batches = 0
    for batch_idx, batch in enumerate(val_loader):
        torch.cuda.empty_cache()

        # Get input data
        image = batch['image'].to(device)
        pc_incomplete = batch['pc_incomplete'].to(device)
        mass = batch['mass'].unsqueeze(1).to(device)
        pc_gt = batch['pc_complete'].to(device)
        is_pc_gt = (len(batch['pc_complete'][0].size()) != 0)

        # Forward pass
        with torch.no_grad():
            preds = model((pc_incomplete, image))
        
        # Get predictions
        pc_reconstructed = None
        if args.pc_completion:
            mass_pred, pc_reconstructed = preds
        else:
            mass_pred = preds
        

        loss_mass = None
        loss_pc = None
        if not is_pc_gt:
            # Compute mass loss (ALDE loss -> Absolute Log Difference Error)
            loss_mass = torch.mean(torch.abs(torch.log(mass_pred) - torch.log(mass)))
            loss_total = loss_mass
        else:
            pc_incomplete = pc_incomplete.squeeze(dim = 1)
            loss_pc = chamfer_distance_bidirectional(pc_gt, pc_reconstructed)
            loss_total = loss_pc

        
        average_loss.update(loss_total.item(), 1)
        if loss_pc is not None:
            average_loss_pc.update(loss_pc.item(), 1)
        if loss_mass is not None:
            average_loss_mass.update(loss_mass.item(), 1)
            computed_result = compute_metrics_mass(mass_pred, mass)

        loss_total = average_loss.avg
        if loss_pc is not None:
            loss_pc = average_loss_pc.avg
        if loss_mass is not None:
            loss_mass = average_loss_mass.avg
        
        logging.progress_bar(batch_idx, val_loader_len, args.epochs, epoch)

        if loss_mass is not None:
            for key in result_metrics.keys():
                result_metrics[key] += computed_result[key]
            mass_batches += 1

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (mass_batches + 1)

    if loss_pc is not None:
        return result_metrics, loss_total, loss_pc, loss_mass
    return result_metrics, loss_total, None, None


if __name__ == '__main__':
    main()
