from scipy.fftpack import fft
from testers import *
import scipy.io as spio
import pickle
import numpy as np
import torch
import yaml
import copy
import os

def accuracy_k(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = min(max(topk), output.shape[1])
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_layer_config(args,model,task):
    # fixed_layer: shared layers for all tasks while config is set to be 0
    
    
    prune_ratios = {}
    pruned_layer = []

    # For fixed layer
    fixed_layer = []
    if args.dataset == 'cifar' or args.dataset == 'mixture':
        fixed_layer = ['module.fc1.bias']
    elif args.dataset == 'mnist':
        fixed_layer = ['module.fc1.bias','module.fc2.bias']

    # For output layer
    output_layer = []
    if args.dataset == 'cifar' or args.dataset == 'mixture':
        output_layer = ['module.fc2.weight','module.fc2.bias']
    elif args.dataset == 'mnist':
        output_layer = ['module.fc3.weight','module.fc3.bias']
    elif args.dataset in ['rfmls', 'dronerc']:
        output_layer = ['ds_module.1.ds1_activate.eta.weight','ds_module.1.ds1_activate.xi.weight', 'ds_module.ds1_activate.xi.weight']
    # For pruned layer
    config_setting = list(map(float, args.config_setting.split(",")))


    if len(config_setting) == 1:
        sparse_setting = float(config_setting[0])
    else:
        sparse_setting = float(config_setting[task])/float(sum(config_setting))

    sparse_setting = 1-sparse_setting*args.config_shrink

    with torch.no_grad():
        for name, W in (model.named_parameters()):
            if args.dataset == 'cifar' or args.dataset == 'mixture':
                if 'weight' in name and name!='module.fc2.weight' and name not in fixed_layer:
                #if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)
            elif args.dataset == 'mnist':
                if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                #if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)
            elif args.dataset == 'rfmls' or 'dronerc':
                if 'weight' in name and 'bn' not in name and 'ds' not in name:
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)

    args.prune_ratios = prune_ratios
    print('Pruned ratio:',sparse_setting)
            
    args.pruned_layer = pruned_layer
    args.fixed_layer = fixed_layer
    args.output_layer = output_layer
    print('Pruned layer:',pruned_layer)
    print('Fixed layer:',fixed_layer)
    print('Output layer:',output_layer)
    return args

def model_loader(args, classes_per_task=None):
    if args.adaptive_mask:
        from models.masknet import CifarNet, MnistNet
        from models.masknet import ResNet50_1d, ResNet18_1d
    else:
        from models.cifarnet import CifarNet
        from models.mnistnet import MnistNet
        from models.masknet import ResNet50_1d, ResNet18_1d

    
    if args.arch == 'cifarnet':
        model = CifarNet(args.input_size, args.classes)
    elif args.arch == 'mnistnet':
        model = MnistNet(args.input_size, args.classes)
    elif args.arch == 'rfnet' and classes_per_task is not None:
        model = ResNet18_1d(args.input_size, args.classes, classes_per_task=classes_per_task)
    elif args.arch == 'rfnet':#and "radar" not in args.base_path.lower():
        # model = ResNet50_1d(512, args.classes)
        # model = ResNet50_1d(args.input_size, args.classes)
        model = ResNet18_1d(args.input_size, args.classes)
    
 
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    
    return model

def mask_joint(args,mask1,mask2):
    '''
    mask1 has more 1 than mask2
    return: new mask with only 0s and 1s
    '''

    masks = copy.deepcopy(mask1)
    if not mask2:
        return mask1
    for name in mask1:
        if name not in args.fixed_layer and name in args.pruned_layer:
            non_zeros1,non_zeros2 = mask1[name], mask2[name]
            non_zeros = non_zeros1 + non_zeros2
            
            # Fake float version of |
            under_threshold = non_zeros < 0.5
            above_threshold = non_zeros > 0.5
            non_zeros[above_threshold] = 1
            non_zeros[under_threshold] = 0
            
            masks[name] = non_zeros
    return masks

def mask_reverse(args, mask):
    mask_reverse = copy.deepcopy(mask)
    for name in mask:
        if name in args.pruned_layer:
            mask_reverse[name] = 1.0-mask[name]
    return mask_reverse

def set_model_mask(model,mask):
    '''
    mask:{non-zero:1 ; zero:0}
    '''
    with torch.no_grad():
        for name, W in (model.named_parameters()):
            if name in mask:
                W.data *= mask[name].cuda()

def get_model_mask(model=None):
    masks = {}
    for name, W in (model.named_parameters()):
        if 'mask' in name:
            continue
        weight = W.cpu().detach().numpy()
        non_zeros = (weight != 0)
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros)
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask
        #print(name,zero_mask.nonzero().shape)
    return masks

def cumulate_model(args, task):
    '''
    Cumulate models for individual task.
    '''
    state_dict = {}
    save_path = os.path.join(args.save_path_exp,'task'+str(task))
    
    #state_dict = torch.load(save_path + "/retrained.pt")
    # Trigger for experiment [leave space for future learning]
    if task < args.tasks-1:
        state_dict = torch.load(save_path + "/retrained.pt")
    else: # for last task
        state_dict = torch.load(save_path +"/{}{}.pt".format(args.arch, args.depth) )
            
    if 0 < task:
        save_path = os.path.join(args.save_path_exp,'task'+str(task-1))
        state_dict_prev = torch.load(save_path + "/cumu_model.pt")
        for name, param in state_dict_prev.items():
            if name in args.pruned_layer:
                state_dict[name].copy_(state_dict[name].data + param.data)
    
    save_path = os.path.join(args.save_path_exp,'task'+str(task))
    torch.save(state_dict, save_path+"/cumu_model.pt")

def set_adaptive_mask(model, reset=False, assign_value='', requires_grad=False):
    for name, W in (model.named_parameters()):
        if 'mask' in name:
            
            # set mask to be one
            if reset:
                weight = W.cpu().detach().numpy()
                W.data = torch.ones(weight.shape).cuda()
            
            # set mask to be given value
            elif assign_value:
                weight_name = name.replace('w_mask', 'weight')
                if weight_name in assign_value:
                    W.data = assign_value[weight_name].cuda()
                
            W.requires_grad = requires_grad
            
def load_state_dict(args, model, state_dict, target_keys=[], masks=[]):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. The keys of :attr:`state_dict` must
    exactly match the keys returned by this module's :func:`state_dict()`
    function.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        masks: set target params to be 1.
    """
    own_state = model.state_dict()
    
    if target_keys:
        for name, param in state_dict.items():
            if name in target_keys:       # changed here
                if name not in own_state:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
                param = param.data
                own_state[name].copy_(param)
    elif masks:
        print('Loading layer...')
        for name, param in state_dict.items():
            if name in args.pruned_layer:     # changed here
                param = param.data
                param_t = own_state[name].data
                mask = masks[name].cuda()
                own_state[name].copy_(param + param_t*mask)
                #print(name)
    else:
        print('Loading layer...')
        for name, param in state_dict.items():
            if name not in own_state:     # changed here
                continue
            param = param.data
            own_state[name].copy_(param)
                
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = min(max(topk), output.shape[1])
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.3 ** (epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


import torch
import torch.nn as nn

# class EvidentialLoss(nn.Module):
#     def __init__(self, num_classes, coeff=0.01):
#         super().__init__()
#         self.num_classes = num_classes
#         self.coeff = coeff

#     def forward(self, outputs, targets, omega, beliefs):
#         # outputs: [batch_size, num_classes] (beliefs)
#         # targets: [batch_size] (long)

#         one_hot = torch.nn.functional.one_hot(targets, self.num_classes).float().to(outputs.device)
#         alpha = outputs + 1
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         loglikelihood = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

#         # KL divergence from Dirichlet to uniform prior
#         kl = self._kl_divergence(alpha)
#         return torch.mean(loglikelihood + self.coeff * kl)

#     def _kl_divergence(self, alpha):
#         K = alpha.size(1)
#         beta = torch.ones_like(alpha)
#         S_alpha = torch.sum(alpha, dim=1, keepdim=True)
#         S_beta = torch.sum(beta, dim=1, keepdim=True)

#         lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
#         lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

#         kl = torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha)), dim=1, keepdim=True)
#         return (kl + lnB + lnB_uni).squeeze()

class EvidentialLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.lmda = 1e-1

    def forward(self, E_preds, targets, beliefs):
        # E_preds: Expected value of decision act E_ν(f_ωk), shape [batch_size, num_classes]
        # targets: Ground truth indices, shape [batch_size]

        # print(torch.sum(E_preds[0,:]))
        yk = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float().to(E_preds.device)

        if torch.isnan(E_preds).any() or torch.isinf(E_preds).any():
            raise ValueError("NaN or Inf in E_preds")
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise ValueError("NaN or Inf in targets")
        
        if torch.isnan(yk).any() or torch.isinf(yk).any():
            raise ValueError("NaN or Inf in yk")

        ## KL divergence on beliefs
        pred_classes = torch.argmax(E_preds, dim=1)
        incorrect = (pred_classes != targets)
        kl_loss = torch.tensor(0.0, device=E_preds.device)
        if incorrect.any():
            alpha = E_preds[incorrect] + 1
            if torch.isnan(alpha).any() or torch.isinf(alpha).any():
                raise ValueError("NaN or Inf in alpha before KL divergence")
            kl_loss = self._kl_divergence(alpha)
            if kl_loss.abs().max() > 1e6:
                print(f"Large loss: {kl_loss}")
            kl_loss = torch.clamp(kl_loss, min=1e-8, max=1)
            kl_loss = torch.mean(kl_loss)

            if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
                raise ValueError("NaN or Inf in kl_loss")

        # L = - ∑ y_k * log(Eν(f_ωk)) + (1 - y_k) * log(1 - Eν(f_ωk))
        log_probs = yk * torch.log(E_preds + 1e-8) + (1 - yk) * torch.log(1 - E_preds + 1e-8)
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            raise ValueError("NaN or Inf in log_probs")
        loss = -torch.sum(log_probs, dim=1)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError("NaN or Inf in loss")

        if loss.abs().max() > 1e6:
            print(f"Large loss: {loss}")
            
        if kl_loss.abs().max() > 1e6:
            print(f"Large loss: {kl_loss}")
        # print(self.lmda * torch.mean(self._kl_divergence(alpha)))
        # kl_loss = 20*kl_loss
        loss = torch.mean(loss) #+ kl_loss
        
        # check_loss_autograd(loss)
        # print(loss, kl_loss)
        return loss

    def _kl_divergence(self, alpha):
            with torch.autograd.set_detect_anomaly(True):
                alpha = torch.clamp(alpha, min=1e-4)
                K = alpha.size(1)
                beta = torch.ones_like(alpha)
                S_alpha = torch.sum(alpha, dim=1, keepdim=True)
                S_beta = torch.sum(beta, dim=1, keepdim=True)

                lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
                lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

                kl = torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha)), dim=1, keepdim=True)
            return (kl + lnB + lnB_uni).squeeze()
    
    def kl_dirichlet_debug(alpha):
        debug = {}

        K = alpha.size(1)

        # Check alpha for NaNs or negatives
        if torch.isnan(alpha).any():
            print("NaN in alpha")
        if (alpha <= 0).any():
            print("Alpha has non-positive values")

        debug['alpha'] = alpha.clone().detach()

        beta = torch.ones_like(alpha)
        debug['beta'] = beta

        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        debug['S_alpha'] = S_alpha
        debug['S_beta'] = S_beta

        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        if torch.isnan(lnB).any():
            print("NaN in lnB")

        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        if torch.isnan(lnB_uni).any():
            print("NaN in lnB_uni")

        debug['lnB'] = lnB
        debug['lnB_uni'] = lnB_uni

        digamma_alpha = torch.digamma(alpha)
        digamma_S_alpha = torch.digamma(S_alpha)

        if torch.isnan(digamma_alpha).any():
            print("NaN in digamma(alpha)")
        if torch.isnan(digamma_S_alpha).any():
            print("NaN in digamma(S_alpha)")

        debug['digamma_alpha'] = digamma_alpha
        debug['digamma_S_alpha'] = digamma_S_alpha

        kl_term = torch.sum((alpha - beta) * (digamma_alpha - digamma_S_alpha), dim=1, keepdim=True)
        if torch.isnan(kl_term).any():
            print("NaN in kl_term")

        debug['kl_term'] = kl_term

        kl = kl_term + lnB + lnB_uni
        if torch.isnan(kl).any():
            print("Final KL divergence is NaN")

        return kl.squeeze(), debug

def check_loss_autograd(loss, name="loss"):
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{name} is not a tensor — it's type {type(loss)}")

    if not loss.requires_grad:
        print(f"{name} does NOT require gradients — it might be detached from the graph!")

    if torch.isnan(loss).any():
        raise ValueError(f" NaN detected in {name}")

    if torch.isinf(loss).any():
        raise ValueError(f" Inf detected in {name}")

    if loss.dtype != torch.float32 and loss.dtype != torch.float16:
        print(f"{name} has unexpected dtype: {loss.dtype}")

    print(f"{name} passed all autograd checks.")

