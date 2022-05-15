import logging
import random

from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.collate import (
    DataCollatorForBiaffine,
    DataCollatorForGlobalPointer,
    DataCollatorForRicon,
    DataCollatorForSoftmaxCrf,
    DataCollatorForSpan, )

logger = logging.getLogger(__name__)


def get_dataloader_and_dataset(
        args,
        tokenizer,
        accelerator=None,
        text_column_name="text",
        label_column_name="ner_span", ):
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir, )
    trains_ds = ds["train"]
    vals_ds = ds["validation"]

    def tokenize_and_align_train_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=False,
            truncation=True, )
        labels = []
        for i, ner_span in enumerate(examples[label_column_name]):
            ner_type = []
            ner_start = []
            ner_end = []

            for t, s, e in zip(ner_span["type"], ner_span["start"],
                               ner_span["end"]):
                try:
                    token_start = tokenized_inputs.char_to_token(i, s)
                    token_end = tokenized_inputs.char_to_token(i, e)
                except:
                    logger.info(f"{s} is not in word_ids!")
                    continue
                if token_start is None or token_end is None:
                    continue
                ner_type.append(int(t))
                ner_start.append(int(token_start))
                ner_end.append(int(token_end))
            if len(ner_type) == 0:
                ner_type = [args.num_labels]
                ner_start = [0]
                ner_end = [0]

            labels.append({
                "class_labels": ner_type,
                "start_position_labels": ner_start,
                "end_position_labels": ner_end,
            })

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_and_align_dev_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=False,
            truncation=True, )
        labels = []
        for i, ner_span in enumerate(examples[label_column_name]):
            ner_type = []
            ner_start = []
            ner_end = []
            for t, s, e in zip(ner_span["type"], ner_span["start"],
                               ner_span["end"]):
                try:
                    token_start = tokenized_inputs.char_to_token(i, s)
                    token_end = tokenized_inputs.char_to_token(i, e)
                except:
                    logger.info(f"{s} is not in word_ids!")
                    continue
                if token_start is None or token_end is None:
                    continue
                ner_type.append(int(t))
                ner_start.append(int(token_start))
                ner_end.append(int(token_end))
            if len(ner_type) == 0:
                ner_type = [args.num_labels]
                ner_start = [0]
                ner_end = [0]

            labels.append({
                "class_labels": ner_type,
                "start_position_labels": ner_start,
                "end_position_labels": ner_end,
            })

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_and_align_dev_labels_99999(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=99999,
            padding=False,
            truncation=False, )
        labels = []
        for i, ner_span in enumerate(examples[label_column_name]):
            ner_type = []
            ner_start = []
            ner_end = []
            for t, s, e in zip(ner_span["type"], ner_span["start"],
                               ner_span["end"]):
                token_start = tokenized_inputs.char_to_token(i, s)
                token_end = tokenized_inputs.char_to_token(i, e)

                ner_type.append(int(t))
                ner_start.append(int(token_start))
                ner_end.append(int(token_end))

            labels.append({
                "class_labels": ner_type,
                "start_position_labels": ner_start,
                "end_position_labels": ner_end,
            })
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        train_dataset = trains_ds.map(
            tokenize_and_align_train_labels,
            batched=True,
            remove_columns=trains_ds.column_names,
            desc="Running tokenizer on train dataset",
            new_fingerprint=f"train-{args.model_type}-{args.method}-{args.max_length}",
        )
        eval_dataset = vals_ds.map(
            tokenize_and_align_dev_labels,
            batched=True,
            remove_columns=vals_ds.column_names,
            desc="Running tokenizer on dev dataset",
            new_fingerprint=f"dev-{args.model_type}-{args.method}", )
        eval_dataset_99999 = vals_ds.map(
            tokenize_and_align_dev_labels_99999,
            batched=True,
            remove_columns=vals_ds.column_names,
            desc="Running tokenizer on dev dataset",
            new_fingerprint=f"dev-{args.model_type}-{args.method}-99999", )
    common_columns = ["input_ids", "attention_mask"]

    if args.model_type == "chinesebert":
        columns = common_columns + ["pinyin_ids", "labels"]
    else:
        columns = common_columns + ["labels"]

    train_dataset.set_format("torch", columns=columns)
    eval_dataset.set_format("torch", columns=columns)

    if args.method == "span":
        collate_cls = DataCollatorForSpan
    elif args.method in ["globalpointer", "efficient_globalpointer"]:
        collate_cls = DataCollatorForGlobalPointer
    elif args.method in ["softmax", "crf"]:
        collate_cls = DataCollatorForSoftmaxCrf
    elif args.method == "biaffine":
        collate_cls = DataCollatorForBiaffine
    elif args.method == "ricon":
        collate_cls = DataCollatorForRicon
    else:
        raise ValueError(
            "method must be in ['span', 'globalpointer', 'efficient_globalpointer', 'softmax', 'crf', 'biaffine']"
        )

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set:")
        for k, v in train_dataset[index].items():
            logger.info(f"{k} = {v}")

    if args.method == "ricon":
        data_collator = collate_cls(
            tokenizer,
            pad_to_multiple_of=(8 if accelerator.use_fp16 else None),
            num_labels=args.num_labels,
            ngram=args.ngram, )
    else:
        data_collator = collate_cls(
            tokenizer,
            pad_to_multiple_of=(8 if accelerator.use_fp16 else None),
            num_labels=args.num_labels, )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers, )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers, )

    return (
        train_dataloader,
        eval_dataloader,
        train_dataset,
        eval_dataset_99999, )
