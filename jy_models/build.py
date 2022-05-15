from .automodel import get_auto_model
from .common import model_type_to_cls
from .layer import (
    BiaffinePostProcess,
    GlobalPointerPostProcess,
    RiconPostProcess,
    SoftmaxCrfPostProcess,
    SpanPostProcess, )


def build_all(args):
    parent_cls, base_cls, tokenizer_cls, config_cls = model_type_to_cls[
        args.model_type]
    tokenizer_name = (args.tokenizer_name if args.tokenizer_name else
                      args.pretrained_model_name_or_path)
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)
    config = config_cls.from_pretrained(args.pretrained_model_name_or_path)
    config.update(dict(classifier_dropout=None, hidden_dropout_prob=0.1))

    if args.method == "span":
        config.update(
            dict(
                soft_label=args.soft_label,
                num_labels=args.num_labels,
                loss_type=args.loss_type, ))
        postprocessor = SpanPostProcess(args)
    elif args.method in ["globalpointer", "efficient_globalpointer"]:
        config.update(
            dict(
                num_labels=args.num_labels,
                globalpointer_head_size=args.globalpointer_head_size,
                use_rope=not args.donnot_use_rope, ))
        postprocessor = GlobalPointerPostProcess(args)
    elif args.method in ["softmax", "crf"]:
        config.update(
            dict(
                num_labels=args.num_labels,
                loss_type=args.loss_type, ))
        postprocessor = SoftmaxCrfPostProcess(args)
    elif args.method == "biaffine":
        config.update(
            dict(
                num_labels=args.num_labels,
                loss_type=args.loss_type,
                biaffine_hidden_size=args.biaffine_hidden_size, ))
        postprocessor = BiaffinePostProcess(args)
    elif args.method == "ricon":
        config.update(
            dict(
                num_labels=args.num_labels,
                ricon_hidden_size=args.ricon_hidden_size,
                multilabel_loss=args.multilabel_loss,
                orth_loss_eof=args.orth_loss_eof,
                aware_loss_eof=args.aware_loss_eof,
                agnostic_loss_eof=args.agnostic_loss_eof,
                combination=args.combination, ))
        postprocessor = RiconPostProcess(args)
    else:
        raise ValueError(
            "method must be in ['span', 'globalpointer', 'efficient_globalpointer', 'softmax', 'crf', 'biaffine', 'ricon']"
        )

    model_cls = get_auto_model(parent_cls, base_cls, args.method)
    model = model_cls.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        config=config,
        cache_dir=args.model_cache_dir, )
    model.config.label2id = args.label2id
    model.config.id2label = args.id2label
    model.config.entity_label2id = args.entity_label2id
    model.config.entity_id2label = args.entity_id2label
    return model, tokenizer, postprocessor
