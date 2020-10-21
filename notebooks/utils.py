from random import randint
import random
from itertools import chain
import numpy as np

def join_lists(x): return list(chain.from_iterable(x))

def reverse(l): return list(reversed(l))

def mask(ent_str):
    tokens = ent_str.strip().split()
    if len(tokens) == 1:
        return '[ %s ]' % tokens[0]
    elif len(tokens) == 2:
        assert tokens[0] == 'the', ent_str
        return '%s [ %s ]' % (tokens[0], tokens[1])
    else:
        assert False, ent_str

def swap_entities(sent, e0='_X', e1='_Z'):
    return sent.replace(e0, 'xx').replace(e1, e0).replace('xx', e1)


class InputExample(object):
    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, position_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.labels = labels
        # self.tc_labels = tc_labels

def rejoin_tokens(tokens):
    out = []
    while len(tokens) > 0:
        token = tokens.pop(0)
        if token in ['[', 'Ġ['] or token in ['(', 'Ġ('] and tokens[0] not in ['[', 'Ġ[']:
            next_token = tokens.pop(0)  # the maksed word
            next_next_token = tokens.pop(0)  # "]" symbol
            assert next_next_token in [']',  'Ġ]', ')',  'Ġ)']
            token, next_next_token = token.replace('Ġ', ''), next_next_token.replace('Ġ', '')
            out.append(token + next_token + next_next_token)
        else:
            out.append(token)
    return out

def process_markers(tokens, marker='*'):
    markers = [marker, 'Ġ' + marker]
    if any(token in markers for token in tokens):
        marked_positions = [i for i, token in enumerate(tokens) if token in markers]
        tokens = [token for token in tokens if token not in markers]
        marked_positions = [p - i for i, p in enumerate(marked_positions)]
        return tokens, marked_positions
    return tokens, []

def process_mask(tokens, tokenizer):
    output_label = []
    for i, token in enumerate(tokens):
        if token.startswith("[") and token.endswith("]") and token not in tokenizer.all_special_tokens:  # masked word
            token = token[1:-1]
            tokens[i] = tokenizer.mask_token
            output_label.append(tokenizer._convert_token_to_id(token))
        else:
            output_label.append(-1)
#     print('in process_mask:', tokens, output_label)
    return tokens, output_label

def process_tag(tokens, tokenizer):
    output_tokens, output_label = [], []
    for i, token in enumerate(tokens):
        if token.startswith("(") and token.endswith(")") and token not in tokenizer.all_special_tokens:  # tagged word
            token = token[1:-1]
            output_label[-1] = tokenizer.tag2id[token]
        else:
            output_tokens.append(token)
            output_label.append(-1)
    # print('in process_tag:', tokens, output_label)
    assert len(output_tokens) == len(output_label), '%d != %d' % (len(tokens), len(output_label))
    return output_tokens, output_label

def convert_example_to_features(example, max_seq_length, tokenizer, max_noise_len=0,
                                has_markers=False, has_tags=False):
    cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b

    if has_markers:
        tokens_a, marked_positions = process_markers(tokens_a)
        marked_positions = [p + 1 for p in marked_positions]
    if has_tags:
        tokens_a, t1_tag = process_tag(tokens_a, tokenizer)
        tag_ids = [-1] + t1_tag + [-1]
    tokens_a, t1_label = process_mask(tokens_a, tokenizer)
    lm_label_ids = [-1] + t1_label + [-1]

    tokens = []
    segment_ids = []
    # XD
    pos_ids = []
    cur_pos_id = randint(0, max_noise_len) if max_noise_len > 0 else 0
    pos_ids.append(cur_pos_id)
    cur_pos_id += 1

    def inc_pos(token):
        nonlocal cur_pos_id
        pos_ids.append(cur_pos_id)
        cur_pos_id += 1
        if max_noise_len > 0 and token == sep_token: # "[SEP]":
            cur_pos_id += randint(0, max_noise_len)

    tokens.append(cls_token)  # ("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
        inc_pos(token)  # XD

    tokens.append(sep_token)  # ("[SEP]")
    segment_ids.append(0)

    # XD
    pos_ids.append(cur_pos_id)
    cur_pos_id += 1
    if max_noise_len > 0:
        cur_pos_id += randint(0, max_noise_len)

    if tokens_b is not None and len(tokens_b) > 0:
        if has_markers:
            t2_marked_positions = process_markers(tokens_b)
            t2_marked_positions = [len(tokens) + p for p in t2_marked_positions]
            marked_positions += t2_marked_positions
        if has_tags:
            tokens_b, t2_tag = process_tag(tokens_b, tokenizer)
            tag_ids += (t2_tag + [-1])
        tokens_b, t2_label = process_mask(tokens_b, tokenizer)
        lm_label_ids += (t2_label + [-1])

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
            inc_pos(token)  # XD

        tokens.append(sep_token)  # ("[SEP]")
        segment_ids.append(1)
        pos_ids.append(cur_pos_id)  # XD

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)
        pos_ids.append(0)  # XD
        if has_tags: tag_ids.append(-1)  # XD

    assert len(input_ids) == max_seq_length, '%d != %d' % (len(input_ids), max_seq_length)
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    assert len(pos_ids) == max_seq_length  # XD
    lm_label_ids = [-100 if i == -1 else i for i in lm_label_ids]
    if has_tags:
        assert len(tag_ids) == max_seq_length, '%d != %d' % (len(tag_ids), max_seq_length)
        tag_ids = [-100 if i == -1 else i for i in tag_ids]

    features = InputFeatures(input_ids=input_ids,
                             attention_mask=input_mask,
                             token_type_ids=segment_ids,
                             position_ids=pos_ids,  # XD
                             # marked_positions=marked_positions, # XD
                             labels=lm_label_ids,
                             )
    if has_tags: featurs.tc_labeles = tag_ids
    if has_markers: features.marked_positions = marked_positions
    if example.guid <= 0:
        print('in convert_example_to_features: features.labels =', features.labels)
    return features

from torch.utils.data.dataset import Dataset
from transformers.data.datasets.glue import Split

class CHILDDataset(Dataset):
    all_lines = {Split.train: None, Split.dev: None, Split.test: None}

    def __init__(self, all_lines, tokenizer, has_markers=False, has_tags=False,
                max_seq_len=None, max_noise_len=0, split_pct=[0.7, 0.3, 0.0], mode=Split.train):
        if isinstance(mode, str): mode = Split[mode]
        if CHILDDataset.all_lines[mode] is None:
            random.shuffle(all_lines)
            n_dev = int(round(len(all_lines) * split_pct[1]))
            n_test = int(round(len(all_lines) * split_pct[2]))
            n_train = len(all_lines) - n_dev - n_test

            def flatten(lines):
                if len(lines) > 0 and type(lines[0]) == list: lines = join_lists(lines)
                return join_lists(lines) if len(lines) > 0 and type(lines[0]) == list else lines

            CHILDDataset.all_lines[Split.train] = flatten(all_lines[:n_train])
            CHILDDataset.all_lines[Split.dev] = flatten(all_lines[n_train: n_train + n_dev])
            CHILDDataset.all_lines[Split.test] = flatten(all_lines[n_train + n_dev:])

        examples = []
        for i, line in enumerate(CHILDDataset.all_lines[mode]):
            t1, t2, is_next_label = self.split_sent(line)
            tokens_a = rejoin_tokens(tokenizer.tokenize(t1))
            tokens_b = rejoin_tokens(tokenizer.tokenize(t2)) if t2 is not None else None
            example = InputExample(guid=i, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)
            examples.append(example)

        if max_seq_len is None:
            max_seq_len = max([len(example.tokens_a) + len(example.tokens_b) + 3
                if example.tokens_b is not None else len(example.tokens_a) + 2
                for example in examples])

        self.features = [convert_example_to_features(example, max_seq_len, tokenizer,
                        has_markers=has_markers, has_tags=has_tags,max_noise_len=max_noise_len)
             for example in examples]

    def split_sent(self, line):
        label = 0
        if "|||" in line:
            t1, t2 = [t.strip() for t in line.split("|||")]
            assert len(t1) > 0 and len(t2) > 0, "%d %d" % (len(t1), len(t2))
        else:
            # assert self.one_sent
            # t1, t2 = line.strip(), None
            t1, t2 = line, None
        return t1, t2, label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


import matplotlib.pyplot as plt

def plot_head_attn(attn, tokens, ax1=None, marked_positions=[]):
    assert attn.size(0) == attn.size(1) == len(tokens)
#     fig = plt.figure(figsize=(4, round(attn.size(0) / 4)))
    if ax1 is None: ax1 = plt.gca()
    for i in range(attn.size(0)):
        for j in range(attn.size(1)):
            if j in [0, attn.size(1) - 1] or attn[i, j].item() < 0.2: continue
            plt.plot([0, 1], [i, j], color='b', alpha=attn[i, j].item())
    ax1.set_xticks([0, 1])
    ax1.set_xlim(0, 1)
    ax1.axes.xaxis.set_visible(False)

    ax2 = ax1.twinx()
    for ax in [ax1, ax2]: # has to duplicate axes to set color of yticklabel
        ax.set_yticks(np.arange(attn.size(0)))
        ax.set_yticklabels(tokens, fontsize=12)
        for i, yticklabel in enumerate(ax.get_yticklabels()):
            if i in marked_positions:
                yticklabel.set_color('r')
        ax.tick_params(length=0)
        ax.set_ylim(attn.size(0) - 1, 0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    plt.show()