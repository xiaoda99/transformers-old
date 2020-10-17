from random import randint
import random
from itertools import chain

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

def swap_entities(sent, e0='X', e1='Z'):
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
        self.labels = labels
        self.position_ids = position_ids

def rejoin_masked_tokens(tokens):
    out = []
    while len(tokens) > 0:
        token = tokens.pop(0)
        if token not in ['[', ']', 'Ġ[', 'Ġ]']:
            out.append(token)
        else:
            assert token in ['[', 'Ġ[']
            next_token = tokens.pop(0)  # the maksed word
            next_next_token = tokens.pop(0)  # "]" symbol
            assert next_next_token in [']',  'Ġ]']
            token, next_next_token = token.replace('Ġ', ''), next_next_token.replace('Ġ', '')
            out.append(token + next_token + next_next_token)
    return out

def process_markers(tokens, marker='*'):
    if marker in tokens or 'Ġ' + marker in tokens:
        marker_positions = [i for i, token in enumerate(tokens) if token in [marker, 'Ġ' + marker]]
        tokens = [token for token in tokens if token != marker]
        marker_positions = [p - i for i, p in enumerate(marker_positions)]
        return marker_positions
    return []
        
def mask_word(tokens, tokenizer):
    output_label = []
    for i, token in enumerate(tokens):
        if token.startswith("[") and token.endswith("]") and token not in tokenizer.all_special_tokens:  # masked word
            token = token[1:-1]
            tokens[i] = tokenizer.mask_token
            output_label.append(tokenizer._convert_token_to_id(token))
        else:
            output_label.append(-1)
#     print('in mask_word:', tokens, output_label)
    return tokens, output_label

def convert_example_to_features(example, max_seq_length, tokenizer, max_noise_len=0):
    cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    
    marker_positions = process_markers(tokens)
    t1_random, t1_label = mask_word(tokens_a, tokenizer)
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
        t2_marker_positions = process_markers(tokens_b)
        t2_marker_positions = [len(tokens) + p for p in t2_marker_positions]
        marker_positions += t2_marker_positions
        t2_random, t2_label = mask_word(tokens_b, tokenizer)
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

    assert len(input_ids) == max_seq_length, '%d != %d' % (len(input_ids), max_seq_length)
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    assert len(pos_ids) == max_seq_length  # XD
    lm_label_ids = [-100 if i == -1 else i for i in lm_label_ids]

    features = InputFeatures(input_ids=input_ids,
                             attention_mask=input_mask,
                             token_type_ids=segment_ids,
                             position_ids=pos_ids,  # XD
                             marker_positions=marker_positions, # XD
                             labels=lm_label_ids)
    if example.guid <= 0:
        print('in convert_example_to_features: features.labels =', features.labels)
    return features

from torch.utils.data.dataset import Dataset
from transformers.data.datasets.glue import Split

class CHILDDataset(Dataset):
    all_lines = {Split.train: None, Split.dev: None, Split.test: None}
    
    def __init__(self, all_lines, tokenizer, max_seq_len=None, max_noise_len=0, split_pct=[0.7, 0.3, 0.0], mode=Split.train):
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
            tokens_a = rejoin_masked_tokens(tokenizer.tokenize(t1))
            tokens_b = rejoin_masked_tokens(tokenizer.tokenize(t2)) if t2 is not None else None
            example = InputExample(guid=i, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)
            examples.append(example)

        if max_seq_len is None:
            max_seq_len = max([len(example.tokens_a) + len(example.tokens_b) + 3
                if example.tokens_b is not None else len(example.tokens_a) + 2
                for example in examples])

        self.features = [convert_example_to_features(example, max_seq_len, tokenizer, max_noise_len=max_noise_len)
             for example in examples]

    def split_sent(self, line):
        label = 0
        if "|||" in line:
            t1, t2 = [t.strip() for t in line.split("|||")]
            assert len(t1) > 0 and len(t2) > 0, "%d %d" % (len(t1), len(t2))
        else:
            # assert self.one_sent
            t1, t2 = line.strip(), None
        return t1, t2, label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]