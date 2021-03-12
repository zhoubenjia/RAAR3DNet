import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from config import Config
from lib import (Architect, DistributedSampler, SearchNetwork)
from lib import IsoGDData as Datasets
from utils import (AvgrageMeter, calculate_accuracy, create_exp_dir,
                   count_parameters_in_MB, load_checkpoint, save_checkpoint, data_prefetcher, load_pretrained_checkpoint)
import torch.distributed as dist
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Place config Congfile!')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--nprocs', type=int, default=1)
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
# parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--SYNC_BN', action='store_true', help='# Since our batch size is small, setting it to speed up network convergence.')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()
args = Config(args)

try:
  if args.resume:
    args.save = os.path.split(args.resume)[0]
  else:
    args.save = './Checkpoints/Search-EXP-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
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

def main_worker(local_rank, nprocs, args):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(local_rank)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed_all(args.seed)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda(local_rank)
  model = SearchNetwork(GESTURE_CLASSES, criterion, local_rank)
  model = model.cuda(local_rank)
  architect = Architect(model, args)

  if local_rank == 0:
    logging.info("args = %s", args)
    logging.info("param size = %fMB", count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  if args.pretrain:
    model = load_pretrained_checkpoint(model, args.pretrain, local_rank)
    arcParames = torch.load(args.pretrain, map_location=lambda storage, loc: storage.cuda(local_rank))
    model.resume_arch_parameters(arcParames['ArcParameters'], local_rank)
    logging.info("Load pre-train model state_dict Done !")

  if args.resume:
    model, optimizer, strat_epoch, best_acc = load_checkpoint(model, optimizer, args.resume, local_rank)
    arcParames = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(local_rank))
    model.resume_arch_parameters(arcParames['ArcParameters'], local_rank)
    logging.info("The network will resume training.")
    logging.info("Start Epoch: {}, Learning rate: {}, Best accuracy: {}".format(strat_epoch, [g['lr'] for g in optimizer.param_groups], round(best_acc, 4)))
  else:
    strat_epoch = 0
    best_acc = 0.0

  torch.cuda.set_device(local_rank)
  torch.distributed.init_process_group(backend='nccl')
  if args.SYNC_BN:
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
  model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler.last_epoch = strat_epoch

  '''
  Here we can divide the training set into 8/2 or 7/3 to get the actual training set and validation set.
  '''
  if args.type == 'M':
    modality = 'rgb'
  elif args.type == 'K':
    modality = 'depth'
  elif args.type == 'F':
    modality = 'flow'
  else:
    raise Exception('Error in load data modality!')
  train_data = Datasets(args, args.data,
                                args.splits + '/{0}_train_lst.txt'.format(modality), modality, 64, 224, phase='train')

  train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, rank=local_rank, shuffle=True)
  train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                            sampler=train_sampler, pin_memory=True, drop_last=True)

  valid_data = Datasets(args, args.data,
                         args.splits + '/{0}_valid_lst.txt'.format(modality), modality, 64, 224, phase='valid')
  valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, rank=local_rank, shuffle=True)
  valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers,
                           sampler=valid_sampler,
                           pin_memory=True, drop_last=True)

  for epoch in range(strat_epoch, args.epochs):
    train_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)

    lr = scheduler.get_lr()[0]
    genotype = model.module.genotype()
    if local_rank == 0:
      logging.info('genotype = %s', genotype)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, local_rank)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, local_rank, epoch, lr)
    scheduler.step()

    if local_rank == 0:
      if valid_acc > best_acc:
        best_acc = valid_acc
        isbest = True
      else:
        isbest = False
      logging.info('train_acc %f', train_acc)
      logging.info('valid_acc %f', valid_acc)
      state = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'ArcParameters': model.module.arch_parameters(), 'bestacc': best_acc}
      save_checkpoint(state, is_best=isbest, save=args.save)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, local_rank):
  model.train()
  loss_avg = AvgrageMeter()
  arc_loss_avg = AvgrageMeter()
  Acc_avg = AvgrageMeter()
  data_time = AvgrageMeter()
  prefetcher = data_prefetcher(valid_queue)
  input_search, target_search = prefetcher.next()
  end = time.time()
  for step, (input, target) in enumerate(train_queue):
    data_time.update(time.time() - end)
    n = input.size(0)
    input, target = map(lambda x: x.cuda(local_rank, non_blocking=True), [input, target])
    if epoch >= args.warmUp:
      while input_search is not None:
        arc_loss = architect.step(input_search.cuda(local_rank, non_blocking=True), target_search.cuda(local_rank, non_blocking=True))
        arc_loss_avg.update(arc_loss.item(), input_search.size(0))
        input_search, target_search = prefetcher.next()
        break
    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
    optimizer.step()

    accuracy = calculate_accuracy(logits, target)

    torch.distributed.barrier()
    reduced_loss = reduce_mean(loss, args.nprocs)
    reduced_acc = reduce_mean(accuracy, args.nprocs)

    loss_avg.update(reduced_loss.item(), n)
    Acc_avg.update(reduced_acc.item(), n)

    if step % args.report_freq == 0 and local_rank == 0:
      logging.info('epoch:%d, mini-batch:%3d, data time: %.5f, lr = %.5f, loss_CE = %.5f, loss_ARC = %.5f, Accuracy = %.4f' % (
        epoch + 1, step + 1, data_time.avg, lr, loss_avg.avg, arc_loss_avg.avg, Acc_avg.avg))
    end = time.time()
  return Acc_avg.avg, loss_avg.avg

def infer(valid_queue, model, criterion, local_rank, epoch, lr):
  loss_avg = AvgrageMeter()
  Acc_avg = AvgrageMeter()
  infer_time = AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda(local_rank, non_blocking=True)
      target = target.cuda(local_rank, non_blocking=True)
      end = time.time()
      logits = model(input)
      infer_time.update(time.time() - end)
      loss = criterion(logits, target)

      accuracy = calculate_accuracy(logits, target)
      n = input.size(0)

      torch.distributed.barrier()
      reduced_loss = reduce_mean(loss, args.nprocs)
      reduced_acc = reduce_mean(accuracy, args.nprocs)
      loss_avg.update(reduced_loss.item(), n)
      Acc_avg.update(reduced_acc.item(), n)

      if step % args.report_freq == 0 and local_rank == 0:
        logging.info('epoch: %d, mini-batch: %3d, inference time: %.4f, lr = %.5f, loss_CE= %.5f, Accuracy= %.4f' % (
          epoch + 1, step + 1, infer_time.avg, lr, loss_avg.avg, Acc_avg.avg))
  return Acc_avg.avg, loss_avg.avg

if __name__ == '__main__':
  try:
    main_worker(args.local_rank, args.nprocs, args)
  except KeyboardInterrupt:
    if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
      print('remove ‘{}’: Directory'.format(args.save))
      os.system('rm -r {}'.format(args.save))
    os._exit(0)
  except Exception:
    print(traceback.print_exc())
    if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
      print('remove ‘{}’: Directory'.format(args.save))
      os.system('rm -r {}'.format(args.save))
    os._exit(0)
