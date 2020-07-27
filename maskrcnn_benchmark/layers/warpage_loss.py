import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _C

# TODO: Use JIT to replace CUDA implementation in the future.
class _WarpageLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, IOU, gamma, alpha, beta1, beta2):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        ctx.beta1 = beta1
        ctx.beta2 = beta2
        ctx.IOU = IOU
        losses = _C.warpageloss_forward(
            logits, targets, IOU, num_classes, gamma, alpha, beta1, beta2
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        beta1 = ctx.beta1
        beta2 = ctx.beta2
        IOU = ctx.IOU
        d_loss = d_loss.contiguous()
        d_logits = _C.warpageloss_backward(
            logits, targets, d_loss, IOU, num_classes, gamma, alpha, beta1, beta2
        )
        return d_logits, None, None, None, None


warpage_loss_cuda = _WarpageLoss.apply


# def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
#     num_classes = logits.shape[1]
#     gamma = gamma[0]
#     alpha = alpha[0]
#     dtype = targets.dtype
#     device = targets.device
#     class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)
#
#     t = targets.unsqueeze(1)
#     p = torch.sigmoid(logits)
#     term1 = (1 - p) ** gamma * torch.log(p)
#     term2 = p ** gamma * torch.log(1 - p)
#     return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class WarpageLoss(nn.Module):
    def __init__(self, gamma, alpha, beta1, beta2):
        super(WarpageLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, logits, targets, IOU):
        device = logits.device
        if logits.is_cuda:
            loss_func = warpage_loss_cuda
        else:
            print('cpu training unfinished')
            #loss_func = sigmoid_focal_loss_cpu

        loss = loss_func(logits, targets, IOU, self.gamma, self.alpha, self.beta1, self.beta2)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
