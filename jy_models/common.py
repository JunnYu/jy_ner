from chinesebert import ChineseBertConfig, ChineseBertModel, ChineseBertTokenizerFast
from gau_alpha import (
    GAUAlphaConfig,
    GAUAlphaModel,
    GAUAlphaPreTrainedModel,
    GAUAlphaTokenizerFast, )
from roformer import (
    RoFormerConfig,
    RoFormerModel,
    RoFormerPreTrainedModel,
    RoFormerTokenizerFast, )
from transformers.models.albert import (
    AlbertConfig,
    AlbertModel,
    AlbertPreTrainedModel,
    AlbertTokenizerFast, )
from transformers.models.bert import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizerFast, )
from transformers.models.roberta import (
    RobertaConfig,
    RobertaModel,
    RobertaPreTrainedModel,
    RobertaTokenizerFast, )

model_type_to_cls = {
    # parent_cls, base_model_cls, tokenizer_cls, config_cls
    "albert":
    (AlbertModel, AlbertPreTrainedModel, AlbertTokenizerFast, AlbertConfig),
    "bert": (BertPreTrainedModel, BertModel, BertTokenizerFast, BertConfig),
    "roberta": (
        RobertaPreTrainedModel,
        RobertaModel,
        RobertaTokenizerFast,
        RobertaConfig, ),
    "chinesebert": (
        BertPreTrainedModel,
        ChineseBertModel,
        ChineseBertTokenizerFast,
        ChineseBertConfig, ),
    "roformer": (
        RoFormerPreTrainedModel,
        RoFormerModel,
        RoFormerTokenizerFast,
        RoFormerConfig, ),
    "gau_alpha": (
        GAUAlphaPreTrainedModel,
        GAUAlphaModel,
        GAUAlphaTokenizerFast,
        GAUAlphaConfig, ),
}

INF = 1e4
