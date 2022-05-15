import logging
import math
import time
from pprint import pformat

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from args import parse_args
from datasets import load_dataset, load_metric
from fastcore.all import *
from jy_models.build import build_all
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler
from utils import get_dataloader_and_dataset, try_remove_old_ckpt, write_json

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(args,
             model,
             postprocessor,
             dataloader,
             metric,
             accelerator,
             all_labels=None):
    assert all_labels is not None, "all_labels shouldn't be None"
    model.eval()
    all_predictions = []
    for batch in tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc="Evaluating: ",
            leave=False, ):
        labels = batch.pop("labels")
        outputs = model(**batch)
        if args.method == "crf":
            tags = model.crf.decode(outputs[0], batch["attention_mask"])
            outputs = (tags, ) + outputs
        outputs_gathered = postprocessor(
            accelerator.gather(outputs), None, batch["attention_mask"])
        all_predictions.extend(outputs_gathered)

    eval_metric = metric.compute(
        predictions=all_predictions,
        references=all_labels,
        id2label=args.entity_id2label,
        digits=6, )
    mmm_metric = {
        "micro": eval_metric["micro_f1"],
        "macro": eval_metric["macro_f1"],
        "mean": eval_metric["mean_f1"],
    }
    model.train()

    return mmm_metric, eval_metric


def main():
    args = parse_args()
    with_tracking = not args.stop_tracking
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = (Accelerator(
        log_with=args.log_with, logging_dir=args.output_dir)
                   if with_tracking else Accelerator())
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "training.log"),
                mode="a",
                encoding="utf-8", ),
        ], )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Make output dir
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # (1)  load and process dataset
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir, )
    label_list = ds["train"].features["ner_span"].feature["type"].names
    # total labels
    args.num_labels = num_labels = len(label_list)
    if args.method in ["globalpointer", "efficient_globalpointer"]:
        entity_id2label = id2label = dict(zip(range(num_labels), label_list))
        entity_label2id = label2id = dict(zip(label_list, range(num_labels)))
        offset = 0
    elif args.method in ["span", "biaffine", "ricon"]:
        entity_id2label = id2label = dict(
            zip(range(1, num_labels + 1), label_list))  # 0代表 O
        entity_label2id = label2id = dict(
            zip(label_list, range(1, num_labels + 1)))
        offset = 1
    elif args.method in ["softmax", "crf"]:
        new_label_list = ["O"]
        for lb in label_list:
            new_label_list.append("B-" + lb)
        for lb in label_list:
            new_label_list.append("I-" + lb)
        id2label = dict(zip(range(len(new_label_list)), new_label_list))
        label2id = dict(zip(new_label_list, range(len(new_label_list))))
        entity_id2label = dict(zip(range(1, num_labels + 1),
                                   label_list))  # 0代表 O
        entity_label2id = dict(zip(label_list, range(1, num_labels + 1)))
        offset = 1
    else:
        raise ValueError(
            "method must be in ['span', 'globalpointer', 'efficient_globalpointer', 'softmax', 'crf', 'biaffine', 'ricon']"
        )

    args.label2id = label2id
    args.id2label = id2label
    args.entity_id2label = entity_id2label
    args.entity_label2id = entity_label2id
    args.label_list = label_list

    # (2) build model
    model, tokenizer, postprocessor = build_all(args)
    (
        train_dataloader,
        eval_dataloader,
        train_dataset,
        eval_dataset, ) = get_dataloader_and_dataset(
            args,
            tokenizer=tokenizer,
            accelerator=accelerator,
            text_column_name="text",
            label_column_name="ner_span", )
    all_eval_labels = []
    for data in eval_dataset["labels"]:
        per_label = []
        for t, s, e in zip(
                data["class_labels"],
                data["start_position_labels"],
                data["end_position_labels"], ):
            per_label.append((t + offset, s, e))
        all_eval_labels.append(per_label)

    # (3) load metric
    metric = load_metric("utils/triple_metric.py")
    # (4) set optimizer
    no_decay = ["bias", "LayerNorm.weight", "norm"]
    backbone_weights_decay = []
    backbone_weights_nodecay = []
    task_weights_decay = []
    task_weights_nodecay = []
    for n, p in model.named_parameters():
        # 预训练部分
        if model.base_model_prefix in n:
            if not any(nd in n for nd in no_decay):
                backbone_weights_decay.append(p)
            else:
                backbone_weights_nodecay.append(p)
        else:
            # 下游的
            if not any(nd in n for nd in no_decay):
                task_weights_decay.append(p)
            else:
                task_weights_nodecay.append(p)
    optimizer_grouped_parameters = [
        # 预训练
        {
            "params": backbone_weights_decay,
            "weight_decay": args.weight_decay,
        },
        {
            "params": backbone_weights_nodecay,
            "weight_decay": 0.0,
        },
        # down
        {
            "params": task_weights_decay,
            "weight_decay": args.weight_decay,
            "lr": args.task_learning_rate,
        },
        {
            "params": task_weights_nodecay,
            "weight_decay": 0.0,
            "lr": args.task_learning_rate,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps /
                                          num_update_steps_per_epoch)

    args.num_warmup_steps = (
        math.ceil(args.max_train_steps * args.num_warmup_steps_or_radios)
        if isinstance(args.num_warmup_steps_or_radios, float) else
        args.num_warmup_steps_or_radios)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps, )

    # (5) Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler, ) = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if with_tracking:
        experiment_config = {}
        for k, v in vars(args).items():
            if isinstance(v, (str, int, float)):
                experiment_config[k] = v
            else:
                experiment_config[k] = str(v)
        accelerator.init_trackers("ner_no_trainer", experiment_config)

    # Train!
    total_batch_size = (args.per_device_train_batch_size *
                        accelerator.num_processes *
                        args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # log args
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")
    write_json(vars(args), os.path.join(args.output_dir, "args.json"))

    # Only show the progress bar once on each machine.
    use_progress_bar = False
    if use_progress_bar:
        progress_bar = tqdm(
            range(args.max_train_steps),
            leave=False,
            disable=not accelerator.is_local_main_process,
            desc="Training: ", )
    completed_steps = 0
    starting_epoch = 0
    max_f1 = 0.0
    if with_tracking:
        total_loss, logging_loss = 0.0, 0.0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(
                f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch-", "")) + 1
            completed_steps = starting_epoch * len(train_dataloader)
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step-", ""))
            completed_steps = resume_step
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    tic_train = time.time()
    ckpt_time, eval_time = 0.0, 0.0
    padzero_step = len(str(args.max_train_steps))
    padzero_epoch = len(str(args.num_train_epochs - 1))

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None:
                    if step < resume_step:
                        continue
                    elif step == resume_step:
                        tic_train = time.time()
            outputs = model(**batch)
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs[0]
            # We keep track of the loss at each epoch
            if with_tracking:
                total_loss += loss.item()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step % args.gradient_accumulation_steps == 0 or
                    step == len(train_dataloader) - 1):
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if use_progress_bar:
                    progress_bar.update(1)
                completed_steps += 1

                if with_tracking:
                    if (args.logging_steps > 0 and
                            completed_steps % args.logging_steps == 0):
                        speed = args.logging_steps / (
                            time.time() - tic_train - ckpt_time - eval_time)
                        backbone_lr = lr_scheduler.get_last_lr()[0]
                        task_lr = lr_scheduler.get_last_lr()[-1]
                        log_loss = (
                            total_loss - logging_loss) / args.logging_steps
                        accelerator.log(
                            {
                                "train/backbone_lr": backbone_lr,
                                "train/task_lr": task_lr,
                                "train/loss": log_loss,
                                "train/speed": speed,
                            },
                            step=completed_steps, )

                        format_completed_steps = str(completed_steps).zfill(
                            padzero_step)
                        format_epoch = str(epoch).zfill(padzero_epoch)
                        logger.info(
                            f"Train| epoch:[{format_epoch}/{args.num_train_epochs-1}], step:[{format_completed_steps}/{args.max_train_steps}], "
                            f"speed:{speed:.2f} step/s, "
                            f"loss:{log_loss:.5f}, "
                            f"backbone lr:{backbone_lr:.9f}, task lr:{task_lr:.9f}"
                        )
                        logging_loss = total_loss
                        ckpt_time = 0.0
                        eval_time = 0.0
                        tic_train = time.time()

                # Save the model checkpoint checkpointing_steps
                if isinstance(checkpointing_steps, int):
                    tic_ckpt = time.time()
                    if completed_steps % checkpointing_steps == 0:
                        if args.output_dir is not None:
                            ckpt_dir = os.path.join(args.output_dir, "ckpt")
                            output_dir = os.path.join(
                                ckpt_dir, f"step-{completed_steps}")
                            accelerator.save_state(output_dir)
                            try_remove_old_ckpt(
                                ckpt_dir, prefix="step", topk=args.topk)

                    ckpt_time = time.time() - tic_ckpt

                if (args.save_steps > 0 and completed_steps % args.save_steps
                        == 0) or completed_steps == args.max_train_steps:
                    logger.info(
                        f"********** Evaluate Step {completed_steps} **********"
                    )
                    tic_eval = time.time()
                    eval_M, eval_metric = evaluate(
                        args,
                        model,
                        postprocessor,
                        eval_dataloader,
                        metric,
                        accelerator,
                        all_eval_labels, )
                    # log the dev metric
                    for k, v in eval_metric.items():
                        logger.info(f"{k} = {v}")

                    # log to tensorboard
                    metric_dict = {}
                    for k, v in eval_M.items():
                        if hasattr(v, "f1"):
                            for subkey in vars(v).keys():
                                metric_dict[f"eval/{k}/{subkey}"] = getattr(
                                    v, subkey)
                    accelerator.log(metric_dict, step=completed_steps)

                    # write to txt
                    em = eval_M[args.monitor]
                    f1 = em.f1 if hasattr(em, "f1") else em
                    if f1 >= max_f1:
                        max_f1 = f1
                        eval_results = Path(
                            args.output_dir) / "eval_results.txt"
                        eval_results.write_text(
                            pformat(eval_metric), encoding="utf-8")
                    logger.info("************** Evaluate End *************")
                    # save the model and remove the previous one
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        best_dir = os.path.join(args.output_dir, "best")
                        output_dir = os.path.join(
                            best_dir, f"step-{completed_steps}-micro_f1-{f1}")
                        os.makedirs(output_dir, exist_ok=True)
                        accelerator.unwrap_model(model).save_pretrained(
                            output_dir, save_function=accelerator.save)
                        tokenizer.save_pretrained(output_dir)
                        try_remove_old_ckpt(
                            best_dir, prefix="step", topk=args.topk)

                    eval_time = time.time() - tic_eval

            if completed_steps >= args.max_train_steps:
                return

        if args.checkpointing_steps == "epoch":
            tic_ckpt = time.time()
            if args.output_dir is not None:
                ckpt_dir = os.path.join(args.output_dir, "ckpt")
                output_dir = os.path.join(ckpt_dir, f"epoch-{epoch}")
                accelerator.save_state(output_dir)
                try_remove_old_ckpt(ckpt_dir, prefix="epoch", topk=args.topk)
            ckpt_time = time.time() - tic_ckpt


if __name__ == "__main__":
    main()
