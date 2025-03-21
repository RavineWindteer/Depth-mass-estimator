import argparse


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu') # gpu
        parser.add_argument('--data_path',    type=str, default='./datasets/')
        parser.add_argument('--dataset',      type=str, default='shapenetsem',
                            choices=['shapenetsem', 'amazon', 'inference'])
        parser.add_argument('--use_2_datasets',      type=bool, default=False)
        parser.add_argument('--dataset2',      type=str, default='amazon',
                            choices=['shapenetsem', 'amazon', 'inference'])
        parser.add_argument('--pc_model',      type=str, default='pointnet',
                            choices=['none', 'pointnet', 'dgcnn', 'point_transformer'])
        parser.add_argument('--pc_completion',      type=bool, default=False)
        parser.add_argument('--pc_in_dims',      type=int, default=1024,
                            choices=[1024, 2048])
        parser.add_argument('--pc_out_dims',      type=int, default=1024)
        parser.add_argument('--emb_dims',      type=int, default=512,
                            choices=[512, 1024])
        parser.add_argument('--exp_name',     type=str, default='test')
        parser.add_argument('--batch_size_1',   type=int, default=32) # 12
        parser.add_argument('--batch_size_2',   type=int, default=10) 
        parser.add_argument('--workers',      type=int, default=1) # 8

        return parser
