#%%writefile test.py

import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.vgg import vgg16
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed

set_seed(605)


def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    
    checkpoint_file = args.checkpoint

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for testing')
    #print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    #print(train_tsfm)

    train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    backbone = vgg16()
    classifier = BaseClassifier(nattr=train_set.attr_num)
    model = FeatClassifier(backbone, classifier)
    
    #model.load_state_dict(torch.load(filename))
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    #print(checkpoint_file['state_dicts'])
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dicts"])

    criterion = CEL_Sigmoid(sample_weight)

    param_groups = [{'params': model.module.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.module.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
    
    

    tester(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 )

    


def tester(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

          
        
    valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

    lr_scheduler.step(metrics=valid_loss, epoch=1)

        
    valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

    print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

    print(f'{time_str()}')
    print('-' * 60)

        


if __name__ == '__main__':
    parser = argument_parser()

    args = parser.parse_args()
    print(args)
    main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
