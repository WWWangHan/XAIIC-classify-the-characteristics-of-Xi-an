# -*- coding: utf-8 -*-

import argparse
import os
import pdb
import random
import shutil
import time
import warnings
from collections import OrderedDict
try:
    import moxing as mox
except:
    print('not use moxing')
import torch
import torch.nn as nn
import torch.nn.parallel
from build_net import *
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import models as customized_models
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
from ArcMarginProduct import ArcMarginProduct
from prepare_data import prepare_data_on_modelarts

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
parser.add_argument('-a', '--arch', metavar='ARCH', required=True, choices=model_names)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('--epochs', default=100, type=int, metavar='N')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=5, type=int, metavar='N')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--eval_pth', default='', type=str)
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--world-size', default=-1, type=int)
parser.add_argument('--rank', default=-1, type=int)
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str)
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--last_fc_out', default=256, type=int)
parser.add_argument('--multiprocessing-distributed', action='store_true')
parser.add_argument('--scale_size', type=float, default=32.0, help='scale size')

# These arguments are added for adapting ModelArts
parser.add_argument('--num_classes', required=True, type=int, help='the num of classes which your task should classify')
parser.add_argument('--local_data_root', default='/cache/', type=str)
parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
parser.add_argument('--test_data_url', default='', type=str, help='the test data path')
parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
parser.add_argument('--test_data_local', default='', type=str, help='the test data path on local')
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
parser.add_argument('--train_local', default='', type=str, help='the training output results on local')
parser.add_argument('--tmp', default='', type=str, help='a temporary path on local')
parser.add_argument('--deploy_script_path', default='', type=str)

args, unknown = parser.parse_known_args()

best_acc1 = 0

# for painting acc1 and acc5 and loss
epoch_list = []
loss_list = []
acc1_train_list = []
acc5_train_list = []

param_u = str(args.arch) + " " + str(args.batch_size) + " " + str(args.lr) + " " + str(args.momentum) + " " + str(args.weight_decay)

for i in range(args.epochs):
    epoch_list.append(i)

def paint_acc(epoch_list, acc1_list, acc5_list):
    x_begin = min(epoch_list) - 5
    x_end = max(epoch_list) + 5
    x = np.arange(x_begin, x_end)
    l1 = plt.plot(epoch_list, acc1_list, "r--", label="acc1")
    l5 = plt.plot(epoch_list, acc5_list, "g--", label="acc5")
    plt.plot(epoch_list, acc1_list, "ro-", epoch_list, acc5_list, "g+-")
    max_acc1 = max(acc1_list)
    param_title = "b-acc1: " + str(max_acc1) + param_u  
    plt.title(param_title)
    plt.xlabel("epoch")
    plt.ylabel("acc1/acc5")
    plt.legend()
    plt.show()
    

def paint_loss(epoch_list, loss_list):
    x_begin = min(epoch_list) - 5
    x_end = max(epoch_list) + 5
    x = np.arange(x_begin, x_end)
    l = plt.plot(epoch_list, loss_list, "b--", label="loss")
    plt.plot(epoch_list, loss_list, "b^-")
    min_loss = min(loss_list)
    plt.title("min_loss: " + str(min_loss) + "-Loss During Training")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def main():

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        print("=======================================")

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        print("=======================================")

    '''
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    '''
	
    ngpus_per_node = torch.cuda.device_count()
	
    if False: # if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    '''
    # if args.distributed:
    #     if args.dist_url == "env://" and args.rank == -1:
    #         args.rank = int(os.environ["RANK"])
    #     if args.multiprocessing_distributed:
    #         # For multiprocessing distributed training, rank needs to be the
    #         # global rank among all the processes
    #         args.rank = args.rank * ngpus_per_node + gpu
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)
    '''

    # # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     os.environ['TORCH_MODEL_ZOO'] = '../pre-trained_model/pytorch'
    #     if not mox.file.exists('../pre-trained_model/pytorch/resnet50-19c8e357.pth'):
    #         mox.file.copy('s3://ma-competitions-bj4/model_zoo/pytorch/resnet50-19c8e357.pth',
    #                       '../pre-trained_model/pytorch/resnet50-19c8e357.pth')
    #         print('copy pre-trained model from OBS to: %s success' %
    #               (os.path.abspath('../pre-trained_model/pytorch/resnet50-19c8e357.pth')))
    #     else:
    #         print('use exist pre-trained model at: %s' %
    #               (os.path.abspath('../pre-trained_model/pytorch/resnet50-19c8e357.pth')))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    #     models.resnet50()
    #
    # # 改动的地方
    # num_ftrs = model.fc.in_features
    # # model.fc = nn.Linear(num_ftrs, args.num_classes)
    # model.fc = ArcMarginProduct(num_ftrs, args.num_classe, s=args.scale_size)

    #*****************************************************************************************************************************************
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        # os.environ['TORCH_MODEL_ZOO'] = 'pre-trained_model/pytorch'
        # if not mox.file.exists('pre-trained_model/pytorch/resnet50-19c8e357.pth'):
        #     if False:# if mox.file.copy('s3://ma-competitions-bj4/model_zoo/pytorch/resnet50-19c8e357.pth',
        #                   # '../pre-trained_model/pytorch/resnet50-19c8e357.pth')
        #     print('copy pre-trained model from OBS to: %s success' %
        #           (os.path.abspath('../pre-trained_model/pytorch/resnet50-19c8e357.pth')))
        # else:
        #     print('use exist pre-trained model at: %s' %
        #           (os.path.abspath('pre-trained_model/pytorch/resnet50-19c8e357.pth')))
        # # model = models.__dict__[args.arch](pretrained=True)
        print("==========================================")
        model = make_model(args)

    else:
        print("=> creating model of '{}'".format(args.arch))
        model = customized_models.__dict__[args.arch]()
        customized_models.resnet50()

    # 改动的地方
    # num_ftrs = model.fc.out_features

    # model.fc = nn.Linear(num_ftrs, args.num_classes)
    margin = ArcMarginProduct(args.last_fc_out, args.num_classes)
    #*****************************************************************************************************************************************
    print("model: ", model)
    print("margin: ", margin)
    
    if False:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            margin.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            margin = torch.nn.parallel.DistributedDataParallel(margin, device_ids=[args.gpu])
        else:
            model.cuda()
            margin.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            margin = torch.nn.parallel.DistributedDataParallel(margin)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        margin = margin.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            margin = torch.nn.DataParallel(margin).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': margin.parameters()}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[16, 50], gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        # if os.path.isfile(args.resume):
        if mox.file.exists(args.resume) and (not mox.file.is_directory(args.resume)):
            if args.resume.startswith('s3://'):
                restore_model_name = args.resume.rsplit('/', 1)[1]
                mox.file.copy(args.resume, '/cache/tmp/' + restore_model_name)
                args.resume = '/cache/tmp/' + restore_model_name
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if args.resume.startswith('/cache/tmp/'):
                os.remove(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict1'])
            margin.load_state_dict(checkpoint['state_dict2'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data_local, 'train')
    valdir = os.path.join(args.data_local, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    idx_to_class = OrderedDict()
    
    for key, value in train_dataset.class_to_idx.items():
        idx_to_class[value] = key

    # if args.distributed:
    if False:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.eval_pth != '':
        if mox.file.exists(args.eval_pth) and (not mox.file.is_directory(args.eval_pth)):
            if args.eval_pth.startswith('s3://'):
                model_name = args.eval_pth.rsplit('/', 1)[1]
                mox.file.copy(args.eval_pth, '/cache/tmp/' + model_name)
                args.eval_pth = '/cache/tmp/' + model_name
            print("=> loading checkpoint '{}'".format(args.eval_pth))
            if args.gpu is None:
                checkpoint = torch.load(args.eval_pth)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.eval_pth, map_location=loc)
            if args.eval_pth.startswith('/cache/tmp/'):
                os.remove(args.eval_pth)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.eval_pth, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.eval_pth))

        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        if False:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        exp_lr_scheduler.step()
        # train for one epoch
        train(train_loader, model, margin, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if True:
            acc1 = validate(val_loader, model, margin, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = False
            best_acc1 = max(acc1.item(), best_acc1)
            pth_file_name = os.path.join(args.train_local, 'epoch_%s_%s.pth'
                                         % (str(epoch + 1), str(round(acc1.item(), 3))))
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict1': model.state_dict(),
                    'state_dict2': margin.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'idx_to_class': idx_to_class
                }, is_best, pth_file_name, args)

    if args.epochs >= args.print_freq:
        save_best_checkpoint(best_acc1, args)

    # call paint function here
    paint_acc(epoch_list, acc_train_list, acc5_train_list)
    paint_loss(epoch_list, loss_list)


def train(train_loader, model, margin, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    margin.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        output = margin(output, target)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, margin, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    margin.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            output = margin(output, target)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename, args):
    if not is_best:
        torch.save(state, filename)
        if args.train_url.startswith('s3'):
            mox.file.copy(filename,
                          args.train_url + '/' + os.path.basename(filename))
            os.remove(filename)


def save_best_checkpoint(best_acc1, args):
    best_acc1_suffix = '%s.pth' % str(round(best_acc1, 3))
    pth_files = mox.file.list_directory(args.train_url)
    for pth_name in pth_files:
        if pth_name.endswith(best_acc1_suffix):
            break

    # mox.file可兼容处理本地路径和OBS路径
    if not mox.file.exists(os.path.join(args.train_url, 'model')):
        mox.file.mk_dir(os.path.join(args.train_url, 'model'))

    mox.file.copy(os.path.join(args.train_url, pth_name), os.path.join(args.train_url, 'model/model_best.pth'))
    mox.file.copy(os.path.join(args.deploy_script_path, 'config.json'),
                  os.path.join(args.train_url, 'model/config.json'))
    mox.file.copy(os.path.join(args.deploy_script_path, 'customize_service.py'),
                  os.path.join(args.train_url, 'model/customize_service.py'))
    if mox.file.exists(os.path.join(args.train_url, 'model/config.json')) and \
            mox.file.exists(os.path.join(args.train_url, 'model/customize_service.py')):
        print('copy config.json and customize_service.py success')
    else:
        print('copy config.json and customize_service.py failed')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
