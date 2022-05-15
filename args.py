import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="data/ner.py",
        help="The name of the dataset to use (via the datasets library).", )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        choices=[
            "en_ace2004",
            "en_ace2005",
            "en_genia",
            "en_conll03",
            "zh_china_people_daily",
            "zh_cmeee",
            "zh_cluener",
            "zh_medical",
            "zh_msra",
            "zh_onto4",
            "zh_weibo",
            "zh_cail2021",
            "zh_cner",
        ],
        default="zh_msra", )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ), )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="junnyu/roformer_v2_chinese_char_base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.", )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-5,
        help="Backbone LR: Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--task_learning_rate",
        type=float,
        default=1e-3,
        help="Task LR: Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.02,
        help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for AdamW optimizer.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ], )
    parser.add_argument(
        "--num_warmup_steps_or_radios",
        type=eval,
        default=0.1,
        help="Number of steps or radios for the warmup in the lr scheduler.", )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ner_outputs/outputs/",
        help="Where to store the final model.", )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data_caches",
        help="Where to store the final model.", )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help="Model type to use if training from scratch.",
        choices=[
            "roformer", "bert", "chinesebert", "roberta", "albert", "gau_alpha"
        ], )
    parser.add_argument(
        "--method",
        type=str,
        default="ricon",
        choices=[
            "span",
            "globalpointer",
            "efficient_globalpointer",
            "softmax",
            "crf",
            "biaffine",
            "ricon",
        ],
        help="The method of the task.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="logging_steps.", )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=400,
        help="save_steps.", )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="save_topk.", )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers.", )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--model_cache_dir", default=None, type=str, help="model_cache_dir.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.", )
    parser.add_argument(
        "--stop_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="micro",
        choices=[
            "micro",
            "macro",
            "mean",
        ],
        help="The monitor metric of the task.", )
    parser.add_argument(
        "--log_with",
        type=str,
        default="tensorboard",
        choices=["all", "tensorboard", "wandb", "comet_ml"],
        help="logging utils.", )
    parser = add_span_model_specific_args(parser)
    parser = add_globalpointer_model_specific_args(parser)
    parser = add_ricon_model_specific_args(parser)
    parser = add_biaffine_model_specific_args(parser)

    args = parser.parse_args()

    if args.output_dir is not None:
        model_weights = args.pretrained_model_name_or_path.replace(
            "/", "_").replace(":", "")
        args.output_dir = os.path.join(
            args.output_dir,
            f"{args.method}-{args.model_type}-{model_weights}", )
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def add_span_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument(
        "--loss_type",
        default="ce",
        choices=["ce", "focal", "lsce"],
        type=str, )
    parser.add_argument("--soft_label", action="store_true")
    return parser


def add_biaffine_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--biaffine_hidden_size",
        type=int,
        default=128,
        help="biaffine_hidden_size.", )
    return parser


def add_globalpointer_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--globalpointer_head_size",
        default=64,
        type=int, )
    parser.add_argument("--donnot_use_rope", action="store_true")
    return parser


def add_ricon_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--ricon_hidden_size",
        default=200,
        type=int, )
    parser.add_argument(
        "--ngram",
        type=int,
        default=-1,  # -1 or 0 means dont slide
        help="ngram slide.", )
    parser.add_argument(
        "--orth_loss_eof",
        type=float,
        default=0,
        help="orth_loss_eof.", )
    parser.add_argument(
        "--aware_loss_eof",
        type=float,
        default=1.0,
        help="aware_loss_eof.", )
    parser.add_argument(
        "--agnostic_loss_eof",
        type=float,
        default=1.0,
        help="agnostic_loss_eof.", )
    parser.add_argument(
        "--combination",
        type=str,
        default="x,y",
        help="span combination.", )
    parser.add_argument("--multilabel_loss", action="store_true")
    return parser
