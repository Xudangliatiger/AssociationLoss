import torch
from torch import nn

'''修改于7/14'''
class correlationLoss(nn.Module):
    def forward(self, loc_pred, cls_pred, loc_target, weight=None):

        '''修改于7/14'''
        criterion = nn.MSELoss(reduction='mean')

        pred_left = loc_pred[:, 0]
        pred_top = loc_pred[:, 1]
        pred_right = loc_pred[:, 2]
        pred_bottom = loc_pred[:, 3]

        target_left = loc_target[:, 0]
        target_top = loc_target[:, 1]
        target_right = loc_target[:, 2]
        target_bottom = loc_target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        '''修改于7/14'''
        IOU = (area_intersect + 1.0) / (area_union + 1.0)
        IOU = IOU.detach()
        # losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        cls_score = cls_pred.sigmoid()
        cls_score = cls_score.max(1)[0]

        k = 1
        beta = 1
        losses = beta*criterion(k * IOU, cls_score)


        # # option2, log loss
        # pos_inds_A = torch.nonzero(IOU >= cls_score).squeeze(1)
        # pos_inds_B = torch.nonzero(IOU < cls_score).squeeze(1)
        #
        # K1 = 1
        # K2 = 1.75
        # losses_part_A = (0.25*(1-cls_score[pos_inds_A]).pow(2))*torch.log(cls_score[pos_inds_A]) \
        #                +K1*(- IOU[pos_inds_A]*torch.log(cls_score[pos_inds_A]) + cls_score[pos_inds_A] - IOU[pos_inds_A])
        # losses_part_B = (0.25*(1-cls_score[pos_inds_B]).pow(2))*torch.log(cls_score[pos_inds_B]) \
        #                +K2*( - (1-IOU[pos_inds_B]) * torch.log(1-cls_score[pos_inds_B]) +(1- cls_score[pos_inds_B]) - (1-IOU[pos_inds_B]))
        #
        # losses=torch.cat((losses_part_A,losses_part_B),0)
        #
        #

        return losses