import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForSpan:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        labels = ([feature["labels"] for feature in features]
                  if "labels" in features[0].keys() else None)
        new_features = [{k: v
                         for k, v in f.items() if k != "labels"}
                        for f in features]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt", )
        if labels is None:
            return batch

        batch_ner_start = torch.zeros_like(
            batch["input_ids"], dtype=torch.long)
        batch_ner_end = torch.zeros_like(batch["input_ids"], dtype=torch.long)

        for i, label in enumerate(labels):
            for t, s, e in zip(
                    label["class_labels"],
                    label["start_position_labels"],
                    label["end_position_labels"], ):
                if t == self.num_labels:
                    continue
                batch_ner_start[i, s] = t + 1
                batch_ner_end[i, e] = t + 1
        batch["labels"] = [batch_ner_start, batch_ner_end]

        return batch


@dataclass
class DataCollatorForGlobalPointer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        labels = ([feature["labels"] for feature in features]
                  if "labels" in features[0].keys() else None)
        new_features = [{k: v
                         for k, v in f.items() if k != "labels"}
                        for f in features]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt", )
        if labels is None:
            return batch

        bs, seqlen = batch["input_ids"].shape
        # bs,num_labels,seqlen,seqlen
        batch_labels = torch.zeros(
            bs, self.num_labels, seqlen, seqlen, dtype=torch.long)

        for i, label in enumerate(labels):
            for t, s, e in zip(
                    label["class_labels"],
                    label["start_position_labels"],
                    label["end_position_labels"], ):
                if t == self.num_labels:
                    continue
                batch_labels[i, t, s, e] = 1

        batch["labels"] = batch_labels

        return batch


@dataclass
class DataCollatorForSoftmaxCrf:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        labels = ([feature["labels"] for feature in features]
                  if "labels" in features[0].keys() else None)
        new_features = [{k: v
                         for k, v in f.items() if k != "labels"}
                        for f in features]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt", )
        if labels is None:
            return batch

        batch_labels = torch.zeros_like(batch["input_ids"], dtype=torch.long)

        for i, label in enumerate(labels):
            for t, s, e in zip(
                    label["class_labels"],
                    label["start_position_labels"],
                    label["end_position_labels"], ):
                if t == self.num_labels:
                    continue
                if s < e:
                    batch_labels[i, s:e + 1] = t + 1 + self.num_labels
                batch_labels[i, s] = t + 1

        batch["labels"] = batch_labels

        return batch


@dataclass
class DataCollatorForBiaffine:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        labels = ([feature["labels"] for feature in features]
                  if "labels" in features[0].keys() else None)
        new_features = [{k: v
                         for k, v in f.items() if k != "labels"}
                        for f in features]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt", )
        if labels is None:
            return batch

        bs, seqlen = batch["input_ids"].shape
        batch_span_labels = torch.zeros(bs, seqlen, seqlen, dtype=torch.long)

        for i, label in enumerate(labels):
            for t, s, e in zip(
                    label["class_labels"],
                    label["start_position_labels"],
                    label["end_position_labels"], ):
                if t == self.num_labels:
                    continue
                batch_span_labels[i, s, e] = t + 1

        batch["labels"] = batch_span_labels

        return batch


@dataclass
class DataCollatorForRicon:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
    ngram: Optional[int] = 32

    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        labels = ([feature["labels"] for feature in features]
                  if "labels" in features[0].keys() else None)
        new_features = [{k: v
                         for k, v in f.items() if k != "labels"}
                        for f in features]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt", )
        bs, seqlen = batch["input_ids"].shape

        if self.ngram <= 0:
            triangle_mask = torch.triu(
                torch.ones(seqlen, seqlen), diagonal=0).bool()
        else:
            triangle_mask = (torch.triu(
                torch.ones(seqlen, seqlen), diagonal=0) - torch.triu(
                    torch.ones(seqlen, seqlen), diagonal=self.ngram)).bool()

        batch["triangle_mask"] = triangle_mask

        if labels is None:
            return batch

        batch_span_labels = torch.zeros(bs, seqlen, seqlen, dtype=torch.long)
        for i, label in enumerate(labels):
            for t, s, e in zip(
                    label["class_labels"],
                    label["start_position_labels"],
                    label["end_position_labels"], ):
                if t == self.num_labels:
                    continue
                batch_span_labels[i, s, e] = t + 1

        batch_span_labels = batch_span_labels.masked_select(triangle_mask[
            None, :, :]).reshape(bs, -1)

        batch["labels"] = batch_span_labels
        return batch
