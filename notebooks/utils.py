import random

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
                             labels=lm_label_ids)
    return features
