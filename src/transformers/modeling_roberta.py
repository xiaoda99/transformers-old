# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """


import logging
import warnings
import random
# XD
from collections import OrderedDict
import json
import numpy as np
import matplotlib.pyplot as p20
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .configuration_roberta import RobertaConfig
from .file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from .modeling_outputs import (
    CausalLMOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


ROBERTA_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.RobertaTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
            plain tuple.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning. """, ROBERTA_START_DOCSTRING
)
class RobertaForCausalLM(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:

    Example::

        >>> from transformers import RobertaTokenizer, RobertaLMHeadModel, RobertaConfig
        >>> import torch

        >>> tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        >>> config = RobertaConfig.from_pretrained("roberta-base")
        >>> config.is_decoder = True
        >>> model = RobertaLMHeadModel.from_pretrained('roberta-base', config=config, return_dict=True)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutput(
            loss=lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        tc_labels=None,  # XD
        marked_pos_labels=None,  # XD
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Probe(nn.Module):  # XD
    def __init__(self, input_size, output_size, hidden_size=16):
        super().__init__()
        # self.dense0 = nn.Linear(input_size, hidden_size)
        # self.act_fn = nn.functional.relu
        # self.dropout = nn.Dropout(0.1)
        # self.dense1 = nn.Linear(hidden_size, output_size)
        self.dense1 = nn.Linear(input_size, output_size)

    def forward(self, states):
        # states = self.dense0(states)
        # states = self.act_fn(states)b
        # states = self.dropout(states)
        return self.dense1(states)

class RobertaForProbing(BertPreTrainedModel):  # XD
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        # XD
        self.probes = nn.ModuleDict()
        self.probe_tc_label_idx = 1
        self.num_probe_labels = 2
        self.probe_type = 'per_head'
        self.per_head_probe_pos = 'be1'  # 'mask'
        assert self.probe_type in ['accum', 'per_layer', 'per_head', 'per_head_src']
        self.probe_layers = list(range(10, 24))
        self.probe_position_keys = ['e0', 'e1', 'r0', 'r1', 'be0', 'be1', '?', ',', 'cls', 'sep', 'mask']
        self.n_probe_positions = self.config.num_attention_heads \
            if self.probe_type.startswith('per_head') else len(self.probe_position_keys)
        for i in self.probe_layers:
            for probe_i in range(self.n_probe_positions):
                hidden_size = self.config.hidden_size
                if self.probe_type.startswith('per_head'): hidden_size = hidden_size // self.config.num_attention_heads
                self.probes[json.dumps((i, probe_i))] = Probe(hidden_size, self.num_probe_labels)
        self.roberta.encoder.probe_layers = self.probe_layers
        self.roberta.encoder.probe_type = self.probe_type
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None,
                marked_pos_labels=None, probe_positions=None,  # XD
                head_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                labels=None, tc_labels=None, detach=True,  # XD
                output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # XD
        positions = {}
        assert marked_pos_labels.size(1) == 2 and marked_pos_labels.size(2) == 2, str(marked_pos_labels.size())
        # positions['mark0'] = marked_pos_labels[:, self.probe_tc_label_idx, :1]  # (bsz, 2, 2) -> (bsz, 1)
        # positions['mark1'] = marked_pos_labels[:, self.probe_tc_label_idx, 1:]  # (bsz, 2, 2) -> (bsz, 1)
        positions['e0'] = marked_pos_labels[:, 0, :1]  # (bsz, 2, 2) -> (bsz, 1)
        positions['e1'] = marked_pos_labels[:, 0, 1:]  # (bsz, 2, 2) -> (bsz, 1)
        positions['r0'] = marked_pos_labels[:, 1, :1]  # (bsz, 2, 2) -> (bsz, 1)
        positions['r1'] = marked_pos_labels[:, 1, 1:]  # (bsz, 2, 2) -> (bsz, 1)
        positions['cls'] = (input_ids == self.tokenizer.cls_token_id).nonzero()[:, 1:]
        positions['sep'] = (input_ids == self.tokenizer.sep_token_id).nonzero()[:, 1:]
        # be_positions = torch.ones_like(cls_positions) * 2
        be0_pos, be1_pos = (input_ids == self.tokenizer._convert_token_to_id('Ġis')).nonzero() \
            .view(input_ids.size(0), 2, -1).chunk(2, dim=1)
        positions['be0'], positions['be1'] = be0_pos[:, 0, 1:], be1_pos[:, 0, 1:]
        positions['?'] = (input_ids == self.tokenizer._convert_token_to_id('?')).nonzero()[:, 1:]
        positions[','] = (input_ids == self.tokenizer._convert_token_to_id(',')).nonzero()[:, 1:]
        positions['mask'] = (input_ids == self.tokenizer.mask_token_id).nonzero()[:, 1:]
        positions['mean'] = 'mean'
        self.roberta.encoder.probe_positions = positions[self.per_head_probe_pos] \
            if self.probe_type.startswith('per_head') \
            else torch.cat([positions[k] for k in self.probe_position_keys], dim=-1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0].detach()  # XD
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # XD
        probed_hidden_states = outputs[-1]  # L * (bsz, 4, hidden_size)
        if detach:
            probed_hidden_states = {k: v.detach() for k, v in probed_hidden_states.items()}
        probe_logits = []
        for key, probe in self.probes.items():
            layer_i, probe_i = json.loads(key)
            # print('layer_i, probe_i =', layer_i, probe_i, probed_hidden_states[layer_i].size())
            probe_logits.append(probe(probed_hidden_states[layer_i][:, probe_i]))
        probe_logits = torch.stack(probe_logits, dim=1) # L*4 (bsz, num_probe_labels) -> (bsz, L*4, num_probe_labels)

        probe_loss = None
        if tc_labels is not None:
            loss_fct = CrossEntropyLoss()
            bsz, n_probes, _ = probe_logits.size()
            tc_labels = tc_labels[tc_labels != loss_fct.ignore_index].view(bsz, -1)
            tc_labels = tc_labels[:, self.probe_tc_label_idx] \
                if self.probe_tc_label_idx is not None \
                else (tc_labels[:, 0] == tc_labels[:, 1]).long()
            assert len(tc_labels) == bsz, str(len(tc_labels))
            tc_labels = tc_labels.unsqueeze(-1).expand(-1, n_probes) # (bsz,) -> (bsz, L*4)
            probe_loss = loss_fct(probe_logits.reshape(-1, self.num_probe_labels), tc_labels.reshape(-1))

        if not return_dict:
            output = (prediction_scores, probe_logits) + outputs[2:]
            return ((probe_loss,) + output) if probe_loss is not None else output

        return MaskedLMOutput(
            loss=probe_loss,
            logits=probe_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings("""RoBERTa Model with a `language modeling` head and a `token classification` head. """, ROBERTA_START_DOCSTRING)
class RobertaDoubleHeadsModel(BertPreTrainedModel):  # XD
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        # XD
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = 20
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)  # config.num_labels)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        tc_labels=None,  # XD
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        # XD
        sequence_output = self.dropout(sequence_output)
        tc_logits = self.classifier(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # XD
        masked_tc_loss = None
        if tc_labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_tc_logits = tc_logits.view(-1, self.num_labels)
                active_tc_labels = torch.where(
                    active_loss, tc_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(tc_labels)
                )
                masked_tc_loss = loss_fct(active_tc_logits, active_tc_labels)
            else:
                masked_tc_loss = loss_fct(tc_logits.view(-1, self.num_labels), tc_labels.view(-1))
            masked_lm_loss += masked_tc_loss

        if not return_dict:
            output = (prediction_scores, tc_logits) + outputs[2:]  # XD
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings("""RoBERTa Model with a `language modeling` head and a `token classification` head. """, ROBERTA_START_DOCSTRING)
class RobertaDoubleHeadsModel2(BertPreTrainedModel):  # XD
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        # XD
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_tc_masks = 1
        self.num_tc_labels = 2
        self.classifier = nn.Linear(self.roberta.encoder.holoattn.hidden_size, self.num_tc_masks * self.num_tc_labels)  # config.num_labels)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        tc_labels=None,  # XD
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        # XD
        holoattn_hidden = outputs[2]  # all_attentions
        tc_logits = self.classifier(holoattn_hidden).view(-1, self.num_tc_masks, self.num_tc_labels)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # XD
        masked_tc_loss = None
        if tc_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_tc_loss = loss_fct(tc_logits.view(-1, self.num_tc_labels), tc_labels[tc_labels != -100])
            masked_lm_loss += masked_tc_loss

        if not return_dict:
            output = (prediction_scores, tc_logits) + outputs[2:]  # XD
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings("""RoBERTa Model with a `language modeling` head and a `token classification` head. """, ROBERTA_START_DOCSTRING)
class RobertaDoubleHeadsModel3(BertPreTrainedModel):  # XD
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.rel_head = RobertaRelHead(config)  # XD

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        marked_pos_labels=None,  # XD
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=True, # XD
            output_hidden_states=True, # XD
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        # XD
        all_hidden_states, all_attentions = outputs[2:]
        hidden_states = all_hidden_states[-1]
        # rel_mask = (input_ids == self.tokenizer.mask_token_id) * (labels == -100)
        rel_mask = (input_ids == self.tokenizer.mask_token_id) * (input_ids.roll(1, -1) != self.tokenizer._convert_token_to_id('Ġ?'))
        bsz = input_ids.size(0)
        hidden_states = hidden_states[rel_mask].view(bsz, -1, self.config.hidden_size) # (bsz, n_rels, hidden_size)
        _, attn_probs = zip(*all_attentions)
        attn_probs = torch.cat(attn_probs, dim=1)  # (bsz, H*L, seq_len, seq_len)
        src_pos, tgt_pos = marked_pos_labels[:, :, 0], marked_pos_labels[:, :, 1]  # (bsz, n_rels)
        holoattn = attn_probs[torch.arange(bsz).unsqueeze(-1), :, tgt_pos, src_pos]  # (bsz, n_rels, H*L)
        pred_holoattn, pred_hidden = self.rel_head(hidden_states)
        rel_loss = MSELoss()(pred_holoattn.view(-1), holoattn.detach().view(-1))
        if not self.training and random.random() < 0.01:
            def mean(a): return a.float().mean().item()
            print('rel_loss = %.4f, nonzero_pct = %.3f %.3f %.3f, max = %.3f %.3f' %
                (rel_loss.item(), mean(pred_hidden > 0), mean(holoattn > 0.3), mean(pred_holoattn > 0),
                holoattn.max().item(), pred_holoattn.max().item()))
            rel_labels = labels[rel_mask].view(bsz, -1) # (bsz, n_rels)
            if True and torch.all(rel_labels > 0):  # rel_tag_type == 'word'
                n_rels = holoattn.size(1)
                assert n_rels == 2, str(n_rels)
                fig, axs = plt.subplots(3, n_rels * 2, sharey=False, figsize=(4 * n_rels * 2, 3.5 * 3))
                avg_pred_hiddens = []
                for rel_i in range(n_rels):
                    rel_token_ids = rel_labels[:, rel_i].bincount().nonzero()[:, 0]
                    assert len(rel_token_ids) == 2, str(len(rel_token_ids))
                    for i, rel_token_id in enumerate(rel_token_ids):
                        rel_sample_indices = (rel_labels[:, rel_i] == rel_token_id).nonzero().squeeze(1)
                        H, L = self.config.num_attention_heads, self.config.num_hidden_layers
                        avg_holoattn = holoattn[rel_sample_indices, rel_i].mean(dim=0).view(H, L).t()
                        avg_pred_holoattn = pred_holoattn[rel_sample_indices, rel_i].mean(dim=0).view(H, L).t()
                        rel_token = self.tokenizer._convert_id_to_token(rel_token_id.item()).replace('Ġ', '')
                        xlabel = rel_token + ' ' + str(len(rel_sample_indices))
                        sns.heatmap(avg_holoattn.detach().cpu(), square=True, cbar=True, ax=axs[0, rel_i * 2 + i]).set_xlabel(xlabel)
                        sns.heatmap(avg_pred_holoattn.detach().cpu(), square=True, cbar=True, ax=axs[1, rel_i * 2 + i]).set_xlabel(xlabel)
                        avg_pred_hidden = pred_hidden[rel_sample_indices, rel_i].mean(dim=0, keepdim=True)
                        avg_pred_hiddens.append(avg_pred_hidden)
                avg_pred_hiddens = torch.cat(avg_pred_hiddens, dim=0)
                sns.heatmap(avg_pred_hiddens.detach().cpu(), cbar=True, ax=axs[2, 0])
                plt.show()
        labels[rel_mask] = -100

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        masked_lm_loss = rel_loss #* 100.  # XD

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]  # XD
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaRelHead(nn.Module):  # XD
    def __init__(self, config):
        super().__init__()
        self.hidden_size = 32
        self.dense = nn.Linear(config.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(0.4 or config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(self.hidden_size, config.num_attention_heads * config.num_hidden_layers)

    def forward(self, features, **kwargs):
        features = self.dropout(features)
        x = self.dense(features)
        x = hidden = torch.relu(x)
        # x = self.layer_norm(x)
        # x = hidden = nn.Softmax(dim=-1)(x)
        x = nn.Dropout(0.2)(x)
        x = self.decoder(x)
        x = torch.relu(x)
        return x, hidden

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


@add_start_docstrings(
    """RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Roberta Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Roberta Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx
