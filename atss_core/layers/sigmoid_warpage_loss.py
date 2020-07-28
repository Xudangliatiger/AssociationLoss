import torch
from torch import nn
import torch.nn.functional as F

class SigmoidWarpageLoss(nn.Module):
    def __init__(self, gamma, alpha, beta):
        super(SigmoidWarpageLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta


    def forward(self, logits, targets):
        '''

        Args:
            logits: samples_number * 80
            targets: samples_number * 2:
                targets[:,0] denotes the label 0-80, where 0 is background and 1-80 is object class;
                targets[:,1] denotes the IoUs, the soft label.

        Returns:

        '''

        #class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
        # pred_left = loc_pred[:, 0]
        # pred_top = loc_pred[:, 1]
        # pred_right = loc_pred[:, 2]
        # pred_bottom = loc_pred[:, 3]
        #
        # target_left = loc_target[:, 0]
        # target_top = loc_target[:, 1]
        # target_right = loc_target[:, 2]
        # target_bottom = loc_target[:, 3]
        #
        # target_aera = (target_left + target_right) * \
        #               (target_top + target_bottom)
        # pred_aera = (pred_left + pred_right) * \
        #             (pred_top + pred_bottom)
        #
        # w_intersect = torch.min(pred_left, target_left) + \
        #               torch.min(pred_right, target_right)
        # h_intersect = torch.min(pred_bottom, target_bottom) + \
        #               torch.min(pred_top, target_top)
        #
        # area_intersect = w_intersect * h_intersect
        # area_union = target_aera + pred_aera - area_intersect
        #
        # '''修改于7/14'''
        # i = (area_intersect+1) / (area_union)
        #t = targets.unsqueeze(1)
        p = torch.sigmoid(logits)

        #c = torch.sigmoid(centerness).unsqueeze(1)
        #p = p * c


        # i = ious
        # I 0-1 normalization
        # i = i/i.max()
        # i_avg = i.avg()

        #
        targets = targets[:,0] - 1
        label_ind_1 = (targets[:,0] >= 0).nonzero().squeeze(1)
        label_ind_2 = targets[label_ind_1,0].long()



        label = p.new_zeros(p.shape)
        label[label_ind_1,label_ind_2] = targets[:,1]
        label = label/label.max()
        pos_ind = ( p <= label).nonzero().squeeze(1)

        zeros = p.new_zeros(p.shape)
        ones = p[pos_ind].new_ones(p[pos_ind].shape)
        # term1 = (p - label).pow(2) * F.binary_cross_entropy_with_logits(
        #    logits, label, reduction='none')
        term1 =  F.binary_cross_entropy_with_logits(logits, zeros, reduction='none') * ( - label + 1) + (label-p)
        term1 = term1 * 0.75


        # p_pos = logits[a,b]
        # p_pos_ot = self.op(p_pos,i)


        # term1[a, b] = (p[a,b] - ious).pow(2) * F.binary_cross_entropy_with_logits(
        #    logits[a,b], ious, reduction='none')
        term1[pos_ind] =  F.binary_cross_entropy_with_logits(logits[pos_ind], ones, reduction='none') * label[pos_ind] \
        + (p[pos_ind] - label[pos_ind])
        term1[pos_ind] = term1[pos_ind] * 0.25

        #term0 = (1 - p) ** gamma * torch.log(p)
        #term0 = torch.log(p)
        #term1 = i * torch.log(p) + i - p        # if   p < I
                                          # when p = I, gradient equals to 0, so we can ignore it
        #term2 = (1 - i) * torch.log(1 - p) + p - i  # if   p > I, p*= 0
        #term3 = (p) ** gamma * torch.log(1 \ p)    # if   p*= 0
        #term3 = torch.log(1 - p)

        # GFL
        # term4 = (p - i)**2 * (i * torch.log(p) + (1 - i) * torch.log(1 - p))

        # weight0 = (t == class_range).float()
        # weight1 = ((t == class_range) * torch.lt(p,i)).float()# if   p < I
        # weight2 = ((t == class_range) * torch.gt(p,i)).float()# if   p > I
        # ? 这里什么意思
        # weight3 = ((t != class_range) * (t >= 0)).float()     # if   p *= 0

        #loss0 = - weight0 * term0 * (alpha)
        # loss1 = - weight1 * term1 * (alpha)
        # loss2 = - weight2 * term2 * (1-alpha)
        # loss3 = - weight3 * term2 * (1-alpha)
        #loss4 = - weight3 * term3 * (1-alpha)

        #temp0 = loss0.sum()
        # temp1 = loss1.sum()
        # temp2 = loss2.sum()
        # temp3 = loss3.sum()
        #temp4 = loss4.sum()

        # loss = - term4.sum()
        # loss = - mask0 * term0 * alpha - mask3 * term3 * (1-alpha)
        sum1 = term1[pos_ind].sum()
        sum2 = term1.sum()
        return term1.sum()

    def ot(self,a,b):

        _, index = a.sort(0,descending=1)
        _, index = index.sort(0)
        b_sor, _ = b.sort(0)

        a_ot = a * (b_sor[index]/a.detach())

        return a_ot

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ", beta=" + str(self.beta)
        tmpstr += ")"
        return tmpstr