import torch
from torch import nn


class SigmoidWarpageLoss(nn.Module):
    def __init__(self, alpha, gamma, beta ):
        super(SigmoidWarpageLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta


    # pred_cls, centerness_flatten_, labels_flatten.int(), labels_IoUs_flatten
    def forward(self, logits, targets, centerness, ious):
        num_classes = logits.shape[1]
        gamma = self.alpha
        alpha = self.gamma
        beta = self.beta

        dtype = targets.dtype
        device = targets.device
        class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

        t = targets.unsqueeze(1)
        p = torch.sigmoid(logits)
        c = torch.sigmoid(centerness)
        p = p*c
        i = ious

        term1 = i * torch.log(p) + i - p            # if   p < I
                                                    # when p = I, gradient equals to 0, so we can ignore it
        term2 = (1 - i) * torch.log(1 - p) + p - i  # if   p > I, p*= 0

#       term3 = p ** gamma * torch.log(1 - p)       # if   p*= 0

        mask1 = ((t == class_range) * torch.lt(p,i)).float()# if   p < I
        mask2 = ((t == class_range) * torch.gt(p,i)).float()# if   p > I
        mask3 = ((t != class_range) * (t >= 0)).float()     # if   p*= 0

        loss = - mask1 * term1 * beta\
               - mask2 * term2 * (1-beta)\
               - mask3 * term2 * (1-beta)

        return loss.sum()


    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ", beta=" + str(self.beta)
        tmpstr += ")"
        return tmpstr