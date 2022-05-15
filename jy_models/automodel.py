import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import (
    CRF,
    Biaffine,
    EfficientGlobalPointer,
    EndpointSpanExtractor,
    FocalLoss,
    GlobalPointer,
    LabelSmoothingCrossEntropy,
    PoolerEndLogits,
    PoolerStartLogits,
    SelfAttentiveSpanExtractor,
    globalpointer_loss,
    multilabel_categorical_crossentropy, )


def get_auto_model(parent_cls, base_cls, method="globalpointer"):
    exist_add_pooler_layer = parent_cls.base_model_prefix in ["bert"]
    if method == "ricon":

        class AutoModelWithRICON(parent_cls):
            def __init__(self, config):
                super().__init__(config)
                self.num_labels = config.num_labels + 1
                self.orth_loss_eof = config.orth_loss_eof
                self.aware_loss_eof = config.aware_loss_eof
                self.agnostic_loss_eof = config.agnostic_loss_eof
                if exist_add_pooler_layer:
                    setattr(
                        self,
                        self.base_model_prefix,
                        base_cls(
                            config, add_pooling_layer=False), )
                else:
                    setattr(self, self.base_model_prefix, base_cls(config))

                hidden_size = config.ricon_hidden_size
                # Regularity-aware module
                self.regularity_aware_lstm = nn.LSTM(
                    input_size=config.hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.4, )

                self.reg = nn.Linear(2 * hidden_size, 2 * hidden_size)
                self.self_attentive_span_extractor = SelfAttentiveSpanExtractor(
                    2 * hidden_size)
                self.endpoint_span_extractor = EndpointSpanExtractor(
                    2 * hidden_size, combination=config.combination)

                # self.reg_linear = nn.Linear(2 * hidden_size, 1)
                # self.aware_biaffine = Biaffine(
                #     2 * hidden_size, 2 * hidden_size, bias_x=False, bias_y=False
                # )
                self.u2 = nn.Linear(
                    self.endpoint_span_extractor.get_output_dim(),
                    2 * hidden_size)
                self.u3 = nn.Linear(4 * hidden_size, 1)
                self.type_linear = nn.Linear(2 * hidden_size, self.num_labels)

                # Regularity-agnostic module
                self.regularity_agnostic_lstm = nn.LSTM(
                    input_size=config.hidden_size,
                    hidden_size=hidden_size,
                    bidirectional=True,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.4, )
                self.mlp1 = nn.Sequential(
                    nn.Linear(2 * hidden_size, 2 * hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(2 * hidden_size, 2 * hidden_size), )
                self.mlp2 = nn.Sequential(
                    nn.Linear(2 * hidden_size, 2 * hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(2 * hidden_size, 2 * hidden_size), )
                self.agnostic_biaffine = Biaffine(2 * hidden_size, 1)

                self.multilabel_loss = config.multilabel_loss
                # donot init
                # self.init_weights()

            def forward(
                    self,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    labels=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    **kwargs, ):
                triangle_mask = kwargs.pop("triangle_mask")
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs, )
                sequence_output = outputs[0]

                # (1) Regularity-aware module
                h_aware = self.regularity_aware_lstm(sequence_output)[0]
                bs = h_aware.shape[0]

                # h_reg (equation 5-7)
                # h_reg shape [bs, seqlen, seqlen, 2d]
                # a_aware = self.reg_linear(h_aware)
                # h_reg = []
                # for i in range(seqlen):
                #     for j in range(seqlen):
                #         if i == j:
                #             h_reg.append(h_aware[:, i])
                #         elif j > i:
                #             if j - i + 1 > self.ngram:
                #                 continue
                #             reg_at = a_aware[:, i : j + 1].softmax(1)
                #             h_reg.append((reg_at * h_aware[:, i : j + 1]).sum(1))
                # h_reg = torch.stack(h_reg, dim=1)
                span_indices = triangle_mask.nonzero().unsqueeze(0).expand(
                    bs, -1, -1)
                h_reg = self.self_attentive_span_extractor(
                    h_aware, span_indices=span_indices)

                # h_span (equation 8)
                # h_span shape [bs, seqlen, seqlen, 2d]
                # 不加这个
                # self.aware_biaffine(h_aware, h_aware)
                h_span = self.u2(
                    self.endpoint_span_extractor(
                        h_aware, span_indices)).reshape_as(h_reg)
                # h_span = self.u2(
                #     torch.cat(
                #         [
                #             h_aware[:, None, :, :].expand(-1, seqlen, -1, -1),
                #             h_aware[:, :, None, :].expand(-1, -1, seqlen, -1),
                #         ],
                #         dim=-1,
                #     )
                # )
                # # [bs, ll, 2d]
                # h_span = h_span.masked_select(triangle_mask[None, :, :, None]).reshape(
                #     bs, -1, hidden_size_2
                # )
                # h_sij  (equation 9-10)
                g_sij = torch.sigmoid(
                    self.u3(torch.cat([h_span, h_reg], dim=-1)))
                h_sij = g_sij * h_span + (1 - g_sij) * h_reg

                # aware_output (equation 11)
                aware_output = self.type_linear(h_sij)

                # (2) Regularity-agnostic module
                h_agnostic = self.regularity_agnostic_lstm(sequence_output)[0]
                h_agnostic_head = self.mlp1(h_agnostic)
                h_agnostic_tail = self.mlp2(h_agnostic)
                # h_agnostic_head (equation 15)
                agnostic_output = self.agnostic_biaffine(h_agnostic_head,
                                                         h_agnostic_tail)

                output = (aware_output, agnostic_output)
                loss = None
                if labels is not None:
                    loss = 0.0
                    ##############################################################3
                    # equation (16) l2 norm loss
                    # horth_output shape [bs, seqlen, num_labels]
                    if self.orth_loss_eof > 0:
                        horth_output = torch.matmul(
                            h_aware.transpose(-2, -1), h_agnostic).squeeze(-1)
                        loss += (self.orth_loss_eof *
                                 horth_output.norm(dim=-1).sum(1).mean())

                    ########################################################################
                    # equation (12) cross entropy loss
                    if self.aware_loss_eof > 0:
                        if self.multilabel_loss:
                            aware_loss = multilabel_categorical_crossentropy(
                                aware_output,
                                F.one_hot(
                                    labels, num_classes=self.num_labels), )
                        else:
                            aware_loss = F.cross_entropy(
                                aware_output.transpose(-2, -1), labels)
                        loss += self.aware_loss_eof * aware_loss

                    ########################################################################
                    if self.agnostic_loss_eof > 0:
                        # agnostic_output shape [bs, -1]
                        agnostic_output = agnostic_output.masked_select(
                            triangle_mask[None, :, :, None]).reshape(bs, -1)
                        # agnostic_label shape [bs, -1]
                        agnostic_label = (labels > 0).to(agnostic_output.dtype)
                        # equation (15) binary target (0 or 1)
                        if self.multilabel_loss:
                            agnostic_loss = multilabel_categorical_crossentropy(
                                agnostic_output, agnostic_label)
                        else:
                            agnostic_loss = F.binary_cross_entropy_with_logits(
                                agnostic_output, agnostic_label)
                        loss += self.agnostic_loss_eof * agnostic_loss

                return ((loss, ) + output) if loss is not None else output

        return AutoModelWithRICON
    elif method in ["globalpointer", "efficient_globalpointer"]:
        global_pointer_cls = (EfficientGlobalPointer
                              if method == "efficient_globalpointer" else
                              GlobalPointer)

        class AutoModelWithGlobalpointer(parent_cls):
            def __init__(self, config):
                super().__init__(config)
                self.num_labels = config.num_labels  # 没有特殊类，因为新开出一个维度作为类别类。
                classifier_dropout = (config.classifier_dropout
                                      if config.classifier_dropout is not None
                                      else config.hidden_dropout_prob)
                self.dropout = nn.Dropout(classifier_dropout)
                self.global_pointer = global_pointer_cls(
                    config.hidden_size,
                    heads=config.num_labels,
                    head_size=config.globalpointer_head_size,
                    RoPE=config.use_rope,
                    max_length=config.max_position_embeddings, )
                if exist_add_pooler_layer:
                    setattr(
                        self,
                        self.base_model_prefix,
                        base_cls(
                            config, add_pooling_layer=False), )
                else:
                    setattr(self, self.base_model_prefix, base_cls(config))
                self.init_weights()

            def forward(
                    self,
                    input_ids,
                    attention_mask=None,
                    labels=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    **kwargs, ):
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs, )
                sequence_output = outputs[0]
                sequence_output = self.dropout(sequence_output)
                logits = self.global_pointer(sequence_output)

                loss = None
                if labels is not None:
                    loss = globalpointer_loss(logits, labels)
                output = (logits, ) + outputs[1:]
                return ((loss, ) + output) if loss is not None else output

        return AutoModelWithGlobalpointer

    elif method == "softmax":

        class AutoModelWithSoftmax(parent_cls):
            def __init__(self, config):
                super().__init__(config)
                classifier_dropout = (config.classifier_dropout
                                      if config.classifier_dropout is not None
                                      else config.hidden_dropout_prob)
                self.num_labels = config.num_labels * 2 + 1  # BI 变成两倍， 0特殊类
                self.dropout = nn.Dropout(classifier_dropout)
                self.classifier = nn.Linear(config.hidden_size,
                                            self.num_labels)

                if exist_add_pooler_layer:
                    setattr(
                        self,
                        self.base_model_prefix,
                        base_cls(
                            config, add_pooling_layer=False), )
                else:
                    setattr(self, self.base_model_prefix, base_cls(config))

                assert config.loss_type in ["lsce", "focal", "ce"]
                if config.loss_type == "lsce":
                    self.criterion = LabelSmoothingCrossEntropy()
                elif config.loss_type == "focal":
                    self.criterion = FocalLoss()
                else:
                    self.criterion = nn.CrossEntropyLoss()

                self.init_weights()

            def forward(
                    self,
                    input_ids,
                    attention_mask=None,
                    labels=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    **kwargs, ):
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs, )
                sequence_output = outputs[0]
                sequence_output = self.dropout(sequence_output)
                logits = self.classifier(sequence_output)

                loss = None
                if labels is not None:
                    if attention_mask is not None:
                        active_loss = attention_mask.reshape(-1).bool()
                        active_labels = labels.reshape(-1)[active_loss]
                        active_logits = logits.reshape(
                            -1, self.num_labels)[active_loss]
                        loss = self.criterion(active_logits, active_labels)
                    else:
                        loss = self.criterion(
                            logits.reshape(-1, self.num_labels),
                            labels.reshape(-1))
                output = (logits, ) + outputs[1:]
                return ((loss, ) + output) if loss is not None else output

        return AutoModelWithSoftmax

    elif method == "crf":

        class AutoModelWithCrf(parent_cls):
            def __init__(self, config):
                super().__init__(config)
                classifier_dropout = (config.classifier_dropout
                                      if config.classifier_dropout is not None
                                      else config.hidden_dropout_prob)
                self.num_labels = config.num_labels * 2 + 1  # BI 变成两倍， 0特殊类
                self.dropout = nn.Dropout(classifier_dropout)
                self.classifier = nn.Linear(config.hidden_size,
                                            self.num_labels)
                self.crf = CRF(num_tags=self.num_labels, batch_first=True)
                if exist_add_pooler_layer:
                    setattr(
                        self,
                        self.base_model_prefix,
                        base_cls(
                            config, add_pooling_layer=False), )
                else:
                    setattr(self, self.base_model_prefix, base_cls(config))
                assert config.loss_type in ["lsce", "focal", "ce"]
                if config.loss_type == "lsce":
                    self.criterion = LabelSmoothingCrossEntropy()
                elif config.loss_type == "focal":
                    self.criterion = FocalLoss()
                else:
                    self.criterion = nn.CrossEntropyLoss()

                self.init_weights()

            def forward(
                    self,
                    input_ids,
                    attention_mask=None,
                    labels=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    **kwargs, ):
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs, )
                sequence_output = outputs[0]
                sequence_output = self.dropout(sequence_output)
                logits = self.classifier(sequence_output)

                loss = None
                if labels is not None:
                    loss = -self.crf(emissions=logits,
                                     tags=labels,
                                     mask=attention_mask)
                output = (logits, ) + outputs[1:]
                return ((loss, ) + output) if loss is not None else output

        return AutoModelWithCrf
    elif method == "span":

        class AutoModelWithSpan(parent_cls):
            def __init__(self, config):
                super().__init__(config)
                self.soft_label = config.soft_label
                self.num_labels = config.num_labels + 1  # 0特殊类
                classifier_dropout = (config.classifier_dropout
                                      if config.classifier_dropout is not None
                                      else config.hidden_dropout_prob)
                self.dropout = nn.Dropout(classifier_dropout)
                self.start_fc = PoolerStartLogits(config.hidden_size,
                                                  self.num_labels)
                if self.soft_label:
                    self.end_fc = PoolerEndLogits(
                        config.hidden_size + self.num_labels, self.num_labels)
                else:
                    self.end_fc = PoolerEndLogits(config.hidden_size + 1,
                                                  self.num_labels)

                assert config.loss_type in ["lsce", "focal", "ce"]
                if config.loss_type == "lsce":
                    self.criterion = LabelSmoothingCrossEntropy()
                elif config.loss_type == "focal":
                    self.criterion = FocalLoss()
                else:
                    self.criterion = nn.CrossEntropyLoss()

                if exist_add_pooler_layer:
                    setattr(
                        self,
                        self.base_model_prefix,
                        base_cls(
                            config, add_pooling_layer=False), )
                else:
                    setattr(self, self.base_model_prefix, base_cls(config))
                self.init_weights()

            def forward(
                    self,
                    input_ids,
                    attention_mask=None,
                    labels=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    **kwargs, ):
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs, )
                sequence_output = outputs[0]
                sequence_output = self.dropout(sequence_output)

                start_logits = self.start_fc(sequence_output)

                start_position_labels = None
                end_position_labels = None
                if labels is not None:
                    start_position_labels = labels[0]
                    end_position_labels = labels[1]

                if start_position_labels is not None and self.training:
                    if self.soft_label:
                        bs, seqlen = input_ids.shape
                        label_logits = torch.zeros(
                            bs,
                            seqlen,
                            self.num_labels,
                            device=start_logits.device, )
                        label_logits.scatter_(
                            2, start_position_labels.unsqueeze(2), 1)
                    else:
                        label_logits = end_position_labels.unsqueeze(2).float()
                else:
                    label_logits = start_logits.softmax(dim=-1)
                    if not self.soft_label:
                        label_logits = (label_logits.argmax(-1).unsqueeze(2)
                                        .type_as(start_logits))

                end_logits = self.end_fc(sequence_output, label_logits)
                outputs = (
                    start_logits,
                    end_logits, ) + outputs[1:]

                if (start_position_labels is not None and
                        end_position_labels is not None):
                    start_logits = start_logits.reshape(-1, self.num_labels)
                    end_logits = end_logits.reshape(-1, self.num_labels)
                    active_loss = attention_mask.reshape(-1).bool()
                    active_start_logits = start_logits[active_loss]
                    active_end_logits = end_logits[active_loss]

                    active_start_labels = start_position_labels.reshape(-1)[
                        active_loss]
                    active_end_labels = end_position_labels.reshape(-1)[
                        active_loss]

                    start_loss = self.criterion(active_start_logits,
                                                active_start_labels)
                    end_loss = self.criterion(active_end_logits,
                                              active_end_labels)
                    total_loss = (start_loss + end_loss) / 2
                    outputs = (total_loss, ) + outputs
                return outputs

        return AutoModelWithSpan

    elif method == "biaffine":

        class AutoModelWithBiaffine(parent_cls):
            def __init__(self, config):
                super().__init__(config)
                self.num_labels = config.num_labels + 1  # 0特殊类
                self.lstm = nn.LSTM(
                    input_size=config.hidden_size,
                    hidden_size=config.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.5,
                    bidirectional=True, )
                self.start_layer = nn.Sequential(
                    nn.Linear(
                        in_features=2 * config.hidden_size,
                        out_features=config.biaffine_hidden_size, ),
                    nn.ReLU(), )
                self.end_layer = nn.Sequential(
                    nn.Linear(
                        in_features=2 * config.hidden_size,
                        out_features=config.biaffine_hidden_size, ),
                    nn.ReLU(), )
                self.biaffne_layer = Biaffine(config.biaffine_hidden_size,
                                              self.num_labels)
                if exist_add_pooler_layer:
                    setattr(
                        self,
                        self.base_model_prefix,
                        base_cls(
                            config, add_pooling_layer=False), )
                else:
                    setattr(self, self.base_model_prefix, base_cls(config))

                assert config.loss_type in ["lsce", "focal", "ce"]
                if config.loss_type == "lsce":
                    self.criterion = LabelSmoothingCrossEntropy()
                elif config.loss_type == "focal":
                    self.criterion = FocalLoss()
                else:
                    self.criterion = nn.CrossEntropyLoss()

                self.init_weights()

            def forward(
                    self,
                    input_ids,
                    attention_mask=None,
                    labels=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    **kwargs, ):
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs, )
                sequence_output = outputs[0]
                encoder_rep = self.lstm(sequence_output)[0]
                start_logits = self.start_layer(encoder_rep)
                end_logits = self.end_layer(encoder_rep)

                logits = self.biaffne_layer(start_logits, end_logits)

                loss = None
                if labels is not None:
                    if attention_mask is not None:
                        active_loss = ((attention_mask[:, :, None] &
                                        attention_mask[:, None, :]).reshape(-1)
                                       .bool())
                        span_labels = labels.reshape(-1)[active_loss]
                        span_logits = logits.reshape(
                            -1, self.num_labels)[active_loss]
                        loss = self.criterion(span_logits, span_labels)
                    else:
                        loss = self.criterion(
                            logits.reshape(-1, self.num_labels),
                            labels.reshape(-1))
                output = (logits, ) + outputs[1:]
                return ((loss, ) + output) if loss is not None else output

        return AutoModelWithBiaffine
