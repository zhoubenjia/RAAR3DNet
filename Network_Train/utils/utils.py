import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict

class ClassAcc():
    def __init__(self, GESTURE_CLASSES):
        self.class_acc = dict(zip([i for i in range(GESTURE_CLASSES)], [0]*GESTURE_CLASSES))
        self.single_class_num = [0]*GESTURE_CLASSES
    def update(self, logits, target):
        pred = torch.argmax(logits, dim=1)
        for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
            if p == t:
                self.class_acc[t] += 1
            self.single_class_num[t] += 1
    def result(self):
        return [round(v / (self.single_class_num[k]+0.000000001), 4) for k, v in self.class_acc.items()]
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def adjust_learning_rate(optimizer, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    df = 0.7
    ds = 40000.0
    lr = lr * np.power(df, step / ds)
    # lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(1, 1, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        #n_correct_elems = correct.float().sum().data[0]
        # n_correct_elems = correct.float().sum().item()
    # return n_correct_elems / batch_size
    return correct_k.mul_(1.0 / batch_size)

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best=False, save='./'):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def load_checkpoint(model, model_path, optimizer=None):
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(local_rank))
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    bestacc = checkpoint['bestacc']
    return model, optimizer, epoch, bestacc

def load_pretrained_checkpoint(model, model_path, local_rank=0):
    # params = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(local_rank))['model']
    params = torch.load(model_path, map_location='cpu')['model']
    new_state_dict = OrderedDict()
    for k, v in params.items():
        name = k[7:] if k[:7] == 'module.' else k
        if name not in ['logits.conv3d.weight', 'logits.conv3d.bias']:
            new_state_dict[name] = v
            print(name)
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(new_state_dict, strict=False)
    return model

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter.avg) for meter in self.meters]
        print('\t'.join(entries))


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def FeatureMap2Heatmap(x, feature, heatmaps=None, sk=None):
    feature_first_frame = x[0, :, 1, :, :].cpu()

    inp = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        inp += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    inp = inp.data.numpy()[::-1]

    # feature_first_frame = y[0, :, 1, :, :].cpu()
    # inp1 = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    # for i in range(feature_first_frame.size(0)):
    #     inp1 += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    # inp1 = inp1.data.numpy()[::-1]

    ## feature
    visFeature = []
    for feat in feature:
        # feature_frame = feat[0, :, 1, :, :].cpu()
        # heatmap = torch.zeros(feature_frame.size(1), feature_frame.size(2))
        # for i in range(feature_frame.size(0)):
        #     heatmap += torch.pow(feature_frame[i, :, :], 2).view(feature_frame.size(1),
        #                                                                 feature_frame.size(2))

        Time_heatmap = torch.zeros(feat.size(3), feat.size(4))
        for j in range(feat.size(2)):
            feature_frame = feat[0, :, j, :, :].cpu()
            heatmap = torch.zeros(feature_frame.size(1), feature_frame.size(2))
            for i in range(feature_frame.size(0)):
                heatmap += torch.pow(feature_frame[i, :, :], 2).view(feature_frame.size(1), feature_frame.size(2))
            Time_heatmap += heatmap
        visFeature.append(Time_heatmap.data.numpy()[::-1])

    if heatmaps is not None:
        heatmaps = heatmaps[0, 0, 32, :, :].cpu()
        heatmaps = heatmaps.data.numpy()[::-1]
    if sk is not None:
        sk = sk[0, 0, 32, :, :].cpu()
        sk = sk.data.numpy()[::-1]

    return inp, visFeature, heatmaps, sk