from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..common import INF


class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)
    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
        >>> input = torch.FloatTensor([2, 1, 2, 2, 1])
        >>> input.requires_grad=True
        >>> target = torch.LongTensor([0, 1, 0, 0, 0])
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(
            self,
            smooth: Optional[float]=1e-4,
            square_denominator: Optional[bool]=False,
            with_logits: Optional[bool]=True,
            ohem_ratio: float=0.0,
            alpha: float=0.0,
            reduction: Optional[str]="mean",
            index_label_position=True, ) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self,
                input: Tensor,
                target: Tensor,
                mask: Optional[Tensor]=None) -> Tensor:
        logits_size = input.shape[-1]

        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input)**self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            loss = 1 - ((2 * interection + self.smooth) / (torch.sum(
                torch.square(flat_input, ),
                -1, ) + torch.sum(torch.square(flat_target), -1) + self.smooth)
                        )

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = (F.one_hot(
            target, num_classes=logits_size).float()
                       if self.index_label_position else target.float())
        flat_input = (torch.nn.Softmax(dim=1)(flat_input)
                      if self.with_logits else flat_input)

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0:
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num -
                                        (mask_neg & pos_example).sum())
                keep_num = min(
                    int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = torch.masked_select(
                        flat_input, neg_example.reshape(-1, 1).bool()).reshape(
                            -1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = (torch.argmax(
                        flat_input,
                        dim=1) == label_idx & flat_input[:, label_idx] >=
                            threshold) | pos_example.reshape(-1)
                    ohem_mask_idx = torch.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(
                    flat_input_idx.reshape(-1, 1),
                    flat_target_idx.reshape(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(
                    flat_input_idx.reshape(-1, 1),
                    flat_target_idx.reshape(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.reshape(-1)
        flat_target = target.reshape(-1).float()
        flat_input = torch.sigmoid(
            flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num + 1]
            cond = (flat_input > threshold) | pos_example.reshape(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction="mean", ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = F.log_softmax(output, dim=-1)
        if self.reduction == "sum":
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == "mean":
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_pred,
            target,
            reduction=self.reduction,
            ignore_index=self.ignore_index)


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self,
                 gamma=2,
                 weight=None,
                 reduction="mean",
                 ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = F.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt)**self.gamma * log_pt
        loss = F.nll_loss(
            log_pt,
            target,
            self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index, )
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(
            self,
            gamma_neg=2,
            gamma_pos=0,
            clip=0.05,
            eps=1e-8,
            disable_torch_grad_focal_loss=False, ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


# sparse
def sparse_multilabel_categorical_crossentropy(y_pred,
                                               y_true,
                                               mask_zero=False,
                                               epsilon=1e-7):
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + INF
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=epsilon, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss


def sparse_globalpointer_loss(y_pred, y_true):
    # y_true shape bs, nclass, max_entity_num
    # y_pred bs, nclass, seqlen * seqlen
    loss = sparse_multilabel_categorical_crossentropy(
        y_true, y_pred.flatten(-2, -1), mask_zero=True)
    return loss.sum(1).mean()


def globalpointer_loss(y_pred, y_true):
    n2 = y_pred.size(-1)**2
    y_pred = y_pred.reshape(-1, n2)
    y_true = y_true.reshape(-1, n2)
    y_true = y_true.to(y_pred.dtype)
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * INF
    y_pred_pos = y_pred - (1 - y_true) * INF
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    # bs,nclass,seqlen,seqlen
    return (neg_loss + pos_loss).sum(1).mean()


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = y_pred.reshape(-1, y_pred.size(-1))
    y_true = y_true.reshape(-1, y_true.shape(-1)).to(y_pred.dtype)
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * INF
    y_pred_pos = y_pred - (1 - y_true) * INF
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return (neg_loss + pos_loss).sum(1).mean()


def adaptive_thresholding_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor=None,
        zero_thres: bool=False,
        eps: float=1e-7,
        inf: float=1e4, ) -> torch.Tensor:
    """Adaptive thresholding Loss"""
    # https://github.com/seukgcode/FastRE/blob/18803f2c307c488e3fb9c795ae9854d4e5c270d4/module/utils.py#L69-L82
    # at_loss(logits, labels, mask)
    # logits shape [bs, seqlen, num_relations + 1]
    # labels shape [bs, seqlen, num_relations]
    # mask shape [bs, seqlen, 1]
    # if pad_first: num_relations index is range(1, num_relations+1), thres index is 0
    # else num_relations index is range(0, num_relations), thres index is num_relations
    ###################################################
    # num_relations = 3
    # logits = torch.randn(1, 4, 1 + num_relations)
    # new_logits = torch.cat([logits[...,1:],logits[...,:1]], -1)
    # labels = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]])
    # mask = torch.ones(1, 4, 1)
    # loss1 = adaptive_thresholding_loss(logits, labels, mask, zero_thres=True)
    # loss2 = adaptive_thresholding_loss(new_logits, labels, mask, zero_thres=False)
    # loss1 == loss2
    zeros = torch.zeros_like(labels[..., :1])
    if zero_thres:
        labels = torch.cat([zeros, labels], dim=-1)
        thres_label = torch.zeros_like(labels)
        thres_label[..., 0] = 1.0
    else:
        labels = torch.cat([labels, zeros], dim=-1)
        thres_label = torch.zeros_like(labels)
        thres_label[..., -1] = 1.0

    pos_logits = logits - (1 - labels - thres_label) * inf
    pos_loss = -torch.sum((pos_logits.softmax(-1) + eps).log() * labels,
                          dim=-1)

    neg_logits = logits - labels * inf
    neg_loss = -torch.sum((neg_logits.softmax(-1) + eps).log() * thres_label,
                          dim=-1)
    loss = pos_loss + neg_loss
    if mask is None:
        loss = loss.mean()
    else:
        loss = loss.sum() / mask.sum()
    return loss
