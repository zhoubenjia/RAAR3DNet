import os, random, math
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import traceback
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from utils.visualizer import Visualizer
from config import Config
from lib import IsoGDData, DistributedSampler
from lib import I3D, NI3D, RAAR3D, genotype
import torch.distributed as dist
from utils import (AvgrageMeter, calculate_accuracy, create_exp_dir, print_func,
                   count_parameters_in_MB, load_checkpoint, save_checkpoint, data_prefetcher, ClassAcc,
                   FeatureMap2Heatmap)

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Place config Congfile!')
parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
parser.add_argument('--distp', action='store_true', help='Distribution training!')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--nprocs', type=int, default=1)

parser.add_argument('--show_class_acc', action='store_true', help='whether show the accuracy of each class?')
parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')

parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
args = parser.parse_args()
args = Config(args)
try:
    if args.resume:
        args.save = os.path.split(args.resume)[0]
    else:
        args.save = './Checkpoints/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
except:
    pass
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

GESTURE_CLASSES = args.num_classes


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main(local_rank, nprocs, args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % local_rank)

    if local_rank == 0:
        logging.info("args = %s", args)

    if args.Network == 'NI3D':
        model = NI3D(args, num_classes=GESTURE_CLASSES, genotype=genotype, pretrained=args.pretrain)
    elif args.Network == 'RAAR3D':
        model = RAAR3D(args, num_classes=GESTURE_CLASSES, genotype=genotype, pretrained=args.pretrain)

    model = model.cuda(local_rank)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(local_rank)
    MSELoss = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / args.warm_up_epochs if epoch < args.warm_up_epochs \
        else 0.5 * (math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    if args.resume:
        model, optimizer, strat_epoch, best_acc = load_checkpoint(model, args.resume, optimizer)
        logging.info("The network will resume training.")
        logging.info("Start Epoch: {}, Learning rate: {}, Best accuracy: {}".format(strat_epoch, [g['lr'] for g in
                                                                                                  optimizer.param_groups],
                                                                                    round(best_acc, 4)))
        if args.resumelr:
            for g in optimizer.param_groups: g['lr'] = args.resumelr
    else:
        strat_epoch = 0
        best_acc = 0.0
    scheduler.last_epoch = strat_epoch

    if args.distp:
        '''
    Init distribution train:
    '''
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')

        if args.SYNC_BN and args.nprocs > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda() if torch.cuda.device_count() > 1 else model.cuda()
    if local_rank == 0:
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    '''
  Load Dataset
  '''
    if args.type == 'M':
        modality = 'rgb'
    elif args.type == 'K':
        modality = 'depth'
    elif args.type == 'F':
        modality = 'flow'
    else:
        raise Exception('Error in load data modality!')
    if os.path.split(args.config)[-1] == 'IsoGD.yml':
        Datasets = IsoGDData
    train_data = Datasets(args, args.data,
                          args.splits + '/{0}_train_lst.txt'.format(modality), modality, args.sample_duration,
                          args.sample_size, phase='train')
    valid_data = Datasets(args, args.data,
                          args.splits + '/{0}_val_lst.txt'.format(modality), modality, args.sample_duration,
                          args.sample_size, phase='valid')
    if args.distp:
        train_sampler = DistributedSampler(train_data)
        valid_sampler = DistributedSampler(valid_data)
    else:
        train_sampler, valid_sampler = None, None
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=(train_sampler is None),
                                              sampler=train_sampler, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=False,
                                              sampler=valid_sampler, pin_memory=True, drop_last=True)

    if args.eval_only:
        valid_acc, valid_obj = infer(valid_queue, model, criterion, MSELoss, local_rank, 1)
        logging.info('valid_acc %f', valid_acc)
        return

    for epoch in range(strat_epoch, args.epochs):
        if args.distp:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

        lr = scheduler.get_lr()[0]
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, MSELoss, optimizer, lr, epoch, local_rank)

        valid_acc, valid_obj = infer(valid_queue, model, criterion, MSELoss, local_rank, epoch)

        scheduler.step()

        if local_rank == 0:
            if valid_acc > best_acc:
                best_acc = valid_acc
                isbest = True
            else:
                isbest = False
            logging.info('train_acc %f', train_acc)
            logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)
            state = {'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'bestacc': best_acc}
            save_checkpoint(state, isbest, args.save)

def train(train_queue, model, criterion, MSELoss, optimizer, lr, epoch, local_rank):
    model.train()
    loss_avg = AvgrageMeter()
    Acc_avg = AvgrageMeter()
    data_time = AvgrageMeter()
    end = time.time()
    for step, (inputs, target, heatmap) in enumerate(train_queue):
        data_time.update(time.time() - end)
        inputs, target = map(lambda x: x.cuda(local_rank, non_blocking=True), [inputs, target])
        if args.Network == 'RAAR3D':
            logits, sk, feature = model(inputs)
            loss_mse = MSELoss(sk, heatmap.cuda(local_rank, non_blocking=True))
            loss_ce = criterion(logits, target)
            loss = loss_ce + args.mse_weight * loss_mse
        else:
            logits, feature = model(inputs)
            loss = criterion(logits, target)

        n = inputs.size(0)
        accuracy = calculate_accuracy(logits, target)
        if args.distp:
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc = reduce_mean(accuracy, args.nprocs)
        else:
            reduced_loss, reduced_acc = loss, accuracy

        loss_avg.update(reduced_loss.item(), n)
        Acc_avg.update(reduced_acc.item(), n)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.report_freq == 0 and local_rank == 0:
            log_info = {
                'Epoch': '{}/{}'.format(epoch + 1, args.epochs),
                'Mini-Batch': '{:0>5d}/{:0>5d}'.format(step + 1, len(train_queue.dataset) // (args.batch_size * args.nprocs)),
                'Data time': round(data_time.avg, 4),
                'Lr': lr,
                'Total Loss': round(loss_avg.avg, 3),
                'Acc': round(Acc_avg.avg, 4)
            }
            print_func(log_info)
            if args.Network == 'RAAR3D':
                visual = FeatureMap2Heatmap(inputs, feature, heatmap, sk)
                vis.featuremap('Input', visual[0])
                vis.featuremap('HEATMAP', visual[2])
                vis.featuremap('SK', visual[3])
            else:
                visual = FeatureMap2Heatmap(inputs, feature)
                vis.featuremap('Input', visual[0])
            for i, feat in enumerate(visual[1]):
                vis.featuremap('feature{}'.format(i + 1), feat)
        end = time.time()
    return Acc_avg.avg, loss_avg.avg


@torch.no_grad()
def infer(valid_queue, model, criterion, MSELoss, local_rank, epoch):
    model.eval()
    loss_avg = AvgrageMeter()
    Acc_avg = AvgrageMeter()
    infer_time = AvgrageMeter()
    class_acc = ClassAcc(GESTURE_CLASSES)
    for step, (inputs, target, heatmap) in enumerate(valid_queue):
        n = inputs.size(0)
        inputs, target = map(lambda x: x.cuda(local_rank, non_blocking=True), [inputs, target])
        end = time.time()
        if args.Network == 'RAAR3D':
            logits, sk, feature = model(inputs)
            loss_mse = MSELoss(sk, heatmap.cuda(local_rank, non_blocking=True))
            loss_ce = criterion(logits, target)
            loss = loss_ce + args.mse_weight * loss_mse
        else:
            logits, feature = model(inputs)
            loss = criterion(logits, target)

        infer_time.update(time.time() - end)

        accuracy = calculate_accuracy(logits, target)
        if args.distp:
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc = reduce_mean(accuracy, args.nprocs)
        else:
            reduced_loss, reduced_acc = loss, accuracy
            class_acc.update(logits, target)

        loss_avg.update(reduced_loss.item(), n)
        Acc_avg.update(reduced_acc.item(), n)

        if step % args.report_freq == 0 and local_rank == 0:
            log_info = {
                'Epoch': epoch + 1,
                'Mini-Batch': '{:0>4d}/{:0>4d}'.format(step + 1, len(valid_queue.dataset) // (args.batch_size * args.nprocs)),
                'Inference time': round(infer_time.avg, 4),
                'LossEC': round(loss_avg.avg, 3),
                'Acc': round(Acc_avg.avg, 4)
            }
            print_func(log_info)


    if args.show_class_acc and not args.distp:
        import matplotlib.pyplot as plt
        if not os.path.exists(args.demo_dir): os.makedirs(args.demo_dir)
        with open(args.demo_dir + '/class_acc.txt', 'a')as f:
            txt = str(class_acc.result()) + '\n'
            f.writelines(txt)

        with open(args.demo_dir + '/class_acc.txt', 'r')as f:
            data = [eval(l.strip()) for l in f.readlines()]
        fig, ax = plt.subplots()
        plot_color, plot_shape = ['b', 'g', 'r', 'c', 'm', 'y', 'o'], ['-', '--', '-.', ':']
        for i, d in enumerate(data):
            ax.plot(list(map(str, range(GESTURE_CLASSES))), d,
                    '{}{}'.format(random.choice(plot_color), random.choice(plot_shape)), label='curve{}'.format(i))
        ax.set(xlabel='class', ylabel='Acc',
               title='The accuracy rate of each class.')
        ax.grid()
        ax.legend()
        fig.savefig(args.demo_dir + '/class_num.png')
        logging.info('Save done!')

    return Acc_avg.avg, loss_avg.avg


if __name__ == '__main__':
    vis = Visualizer(args.visname)
    try:
        main(args.local_rank, args.nprocs, args)
    except KeyboardInterrupt:
        if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
            print('remove ‘{}’: Directory'.format(args.save))
            os.system('rm -rf {}'.format(args.save))
        os._exit(0)
    except Exception:
        print(traceback.print_exc())
        if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
            print('remove ‘{}’: Directory'.format(args.save))
            os.system('rm -rf {}'.format(args.save))
        os._exit(0)
