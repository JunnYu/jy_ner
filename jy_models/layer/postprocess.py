# defaultlist
# __version__ = "1.0.0"
# __author__ = 'c0fec0de'
# __author_email__ = 'c0fec0de@gmail.com'
# __description__ = " collections.defaultdict equivalent implementation of list."
# __url__ = "https://github.com/c0fec0de/defaultlist"
import sys

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics.sequence_labeling import get_entities

from ..common import INF


class defaultlist(list):
    def __init__(self, factory=None):
        """
        List extending automatically to the maximum requested length.

        Keyword Args:

            factory: Function called for every missing index.
        """
        self.__factory = factory or defaultlist.__nonefactory

    @staticmethod
    def __nonefactory():
        return None

    def __fill(self, index):
        missing = index - len(self) + 1
        if missing > 0:
            self += [self.__factory() for idx in range(missing)]

    def __setitem__(self, index, value):
        self.__fill(index)
        list.__setitem__(self, index, value)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__getslice(index.start, index.stop, index.step)
        else:
            self.__fill(index)
            return list.__getitem__(self, index)

    def __getslice__(self, start, stop, step=None):  # pragma: no cover
        # python 2.x legacy
        if stop == sys.maxint:
            stop = None
        return self.__getslice(start, stop, step)

    def __normidx(self, idx, default):
        if idx is None:
            idx = default
        elif idx < 0:
            idx += len(self)
        return idx

    def __getslice(self, start, stop, step):
        end = max((start or 0, stop or 0, 0))
        if end:
            self.__fill(end)
        start = self.__normidx(start, 0)
        stop = self.__normidx(stop, len(self))
        step = step or 1
        r = defaultlist(factory=self.__factory)
        for idx in range(start, stop, step):
            r.append(list.__getitem__(self, idx))
        return r

    def __add__(self, other):
        if isinstance(other, list):
            r = self.copy()
            r += other
            return r
        else:
            return list.__add__(self, other)

    def copy(self):
        """Return a shallow copy of the list. Equivalent to a[:]."""
        r = defaultlist(factory=self.__factory)
        r += self
        return r


class SpanPostProcess(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, labels=None, attention_mask=None):
        mask = ~attention_mask.bool()
        start_preds = outputs[0].argmax(-1).masked_fill(mask, 0).tolist()
        end_preds = outputs[1].argmax(-1).masked_fill(mask, 0).tolist()
        # bs,seqlen,lb

        results = []
        for start_pred, end_pred in zip(start_preds, end_preds):
            example_results = []
            for start, start_label in enumerate(start_pred):
                if start_label == 0:
                    continue
                for end, end_label in enumerate(end_pred[start:]):
                    if start_label == end_label:
                        example_results.append(
                            (start_label, start, start + end))
                        break
            results.append(example_results)

        # this is not good!!!!!!!
        if labels is not None:
            new_labels = []
            for start_labels, end_labels in zip(labels[0].tolist(),
                                                labels[1].tolist()):
                example_labels = []
                for start, start_label in enumerate(start_labels):
                    if start_label == 0:
                        continue
                    for end, end_label in enumerate(end_labels[start:]):
                        if start_label == end_label:
                            example_labels.append(
                                (start_label, start, start + end))
                            break
                new_labels.append(example_labels)
            return results, new_labels

        return results


class GlobalPointerPostProcess(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, labels=None, attention_mask=None):
        mask = ~(attention_mask[:, None, :, None] &
                 attention_mask[:, None, None, :]).bool()
        logits = outputs[0].masked_fill(mask, -INF).cpu().numpy()
        logits[:, :, [0, -1], :] -= INF
        logits[:, :, :, [0, -1]] -= INF

        results = defaultlist(list)
        results[logits.shape[0] - 1] = []
        for b, lb, start, end in zip(*np.where(logits > 0)):
            results[b].append((lb, start, end))

        if labels is not None:
            labels = labels.cpu().numpy()
            new_labels = defaultlist(list)
            new_labels[labels.shape[0] - 1] = []
            for b, lb, start, end in zip(*np.where(labels == 1)):
                new_labels[b].append((lb, start, end))

            return results, new_labels

        return results


class SoftmaxCrfPostProcess(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.id2label = {int(k): v for k, v in self.args.id2label.items()}
        self.entity_id2label = {
            int(k): v
            for k, v in self.args.entity_id2label.items()
        }

    @torch.no_grad()
    def forward(self, outputs, labels=None, attention_mask=None):
        # bs,seqlen,num_labels
        if self.args.method == "softmax":
            mask = ~attention_mask.bool()
            preds = outputs[0].argmax(-1).masked_fill(mask, 0).tolist()
        else:  # crf
            preds = outputs[0].squeeze(0).tolist()

        results = []
        for pred_index in preds:

            pred_text = list(map(lambda i: self.id2label[i], pred_index))

            d = list(
                map(
                    lambda text: (self.args.entity_label2id[text[0]], text[1], text[2]),
                    get_entities(pred_text), ))
            results.append(d)

        if labels is not None:
            labels = labels.cpu().tolist()
            new_labels = []
            for label_index in labels:
                label_text = list(map(lambda i: self.id2label[i], label_index))
                d = list(
                    map(
                        lambda text: (
                            self.args.entity_label2id[text[0]],
                            text[1],
                            text[2], ),
                        get_entities(label_text), ))
                new_labels.append(d)

            return results, new_labels

        return results


class BiaffinePostProcess(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, labels=None, attention_mask=None):
        # bs,seq,seq,lb
        mask = ~(attention_mask[:, :, None] &
                 attention_mask[:, None, :]).bool()
        logits = outputs[0].argmax(-1).masked_fill(mask, 0).cpu().numpy()
        # bs,seqlen,seqlen,num_labels
        results = defaultlist(list)
        results[logits.shape[0] - 1] = []
        for b, start, end in zip(*np.where(logits > 0)):
            results[b].append((logits[b, start, end].item(), start, end))

        if labels is not None:
            labels = labels.cpu().numpy()
            new_labels = defaultlist(list)
            new_labels[labels.shape[0] - 1] = []
            for b, start, end in zip(*np.where(labels > 0)):
                new_labels[b].append(
                    (labels[b, start, end].item(), start, end))

            return results, new_labels

        return results


class RiconPostProcess(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, labels=None, attention_mask=None):
        matrix_size = attention_mask.shape[1]
        shaking_idx2matrix_idx = []
        # 转换成原始的二维index
        for a1 in range(matrix_size):
            for b1 in range(matrix_size):
                if b1 >= a1:
                    if b1 - a1 + 1 > self.args.ngram:
                        continue
                    shaking_idx2matrix_idx.append((a1, b1))

        results = defaultlist(list)
        results[outputs[0].shape[0] - 1] = []

        if self.args.multilabel_loss:
            logits = outputs[0].cpu().numpy()
            for b, idx, c in zip(*np.where(logits > 0)):
                if c == 0:
                    continue
                start, end = shaking_idx2matrix_idx[idx]
                results[b].append((c, start, end))
        else:
            logits = outputs[0].argmax(-1).cpu().numpy()
            for b, idx in zip(*np.where(logits > 0)):
                start, end = shaking_idx2matrix_idx[idx]
                results[b].append((logits[b, idx].item(), start, end))

        if labels is not None:
            labels = labels.cpu().numpy()
            new_labels = defaultlist(list)
            new_labels[labels.shape[0] - 1] = []
            for b, start, end in zip(*np.where(labels > 0)):
                new_labels[b].append(
                    (labels[b, start, end].item(), start, end))

            return results, new_labels

        return results
