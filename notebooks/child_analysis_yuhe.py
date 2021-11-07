# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     argv:
#     - python
#     - -m
#     - ipykernel_launcher
#     - -f
#     - '{connection_file}'
#     display_name: Python 3
#     env: null
#     interrupt_mode: signal
#     language: python
#     metadata: null
#     name: python3
# ---

# +
from IPython import get_ipython
# %load_ext autoreload
# %autoreload 2

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

# +
import sys
custom_path = [
'/nas/xd/projects/transformers/notebooks',
'/nas/xd/transformers/src',
'/home/yuhe/Application/snippets'
]

sys.path = custom_path + sys.path
print('\n'.join(sys.path))

# +
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
os.environ['HF_HOME'] = '/raid/xd/.cache/torch'
from types import MethodType
from tqdm import tqdm
from collections import defaultdict, OrderedDict, Counter
from datetime import datetime
from io import StringIO
from itertools import chain
import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers.data.data_collator import DataCollator, default_data_collator
from transformers import AutoConfig, pipeline
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoForCausalLM,
)
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
# from transformers.trainer_utils import EvaluationStrategy

from utils import *

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# -

from functional import seq
from functional.pipeline import Sequence
from fn import _
from collections import namedtuple 

from child_utils import *
from common_utils import *

# +
import openai

# openai.api_key = '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'
openai.api_key = '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'

text = 'i want to query some gpt3 result'
response = openai.Completion.create(engine="davinci", prompt=text, temperature=0.1, max_tokens=10)

print(response.choices[0].text)
# -

models = {}
cache_dir = '/nas/xd/.cache/torch/transformers/'  # for models besides t5-3b/11b
proxies = {'http': '192.168.50.1:1081'} 

model_name = "EleutherAI/gpt-neo-2.7B"
model = GPTNeoForCausalLM.from_pretrained(model_name, proxies=proxies, cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
models[model_name] = model, tokenizer

# +
# model_name = 'roberta-large'
# model_name = 'gpt2-xl'
model_name = 'EleutherAI/gpt-neo-2.7B'
# model_name = 'EleutherAI/gpt-neo-1.3B'
model, tokenizer = models[model_name]

masked_lm = tokenizer.mask_token is not None and len(tokenizer.additional_special_tokens) == 0
if masked_lm:
    mask_token = tokenizer.mask_token  # '<mask>' for roberta
elif len(tokenizer.additional_special_tokens) > 0:
    mask_token = tokenizer.additional_special_tokens[0]  # '<sxtra_id_0>' for t5
else:
    mask_token = ''  # for gpt2
if masked_lm: nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=5)
# -

blocks = model.transformer.h
L, H = model.config.num_layers, model.config.num_heads
hidden_size = model.config.hidden_size
all_attrs, all_embs = defaultdict(dict), defaultdict(dict)


# +
# adapted from attattr
def scaled_input(emb, num_points, baseline=None):
    # shape of emb: (bsz, num_head, seq_len, seq_len)
    assert emb.size(0) == 1
    if baseline is None: baseline = torch.zeros_like(emb)   
    step = (emb - baseline) / num_points
    res = torch.cat([baseline + step * (i + 1) for i in range(num_points)], dim=0)  # XD
    return res, step

# from https://discuss.pytorch.org/t/get-top-k-indices-values-of-all-rows/89354
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    r = tuple(reversed(out))
    return torch.cat([i.unsqueeze(-1) for i in r], dim=-1).cpu().tolist() if type(index) in [torch.Tensor] else r

def h2topk(h, k=4, return_probs=True):
    if not hasattr(h2topk, 'ln') or h2topk.ln.normalized_shape[0] != h.size(-1):
        h2topk.ln = nn.LayerNorm(h.size(-1))
#     r = model.lm_head(h2topk.ln(h))
    r = model.lm_head(h)
    if return_probs: r = r.softmax(-1)
    return r.topk(k, dim=-1) if k > 0 else r

def globalize(tensor):
    if tensor.dim() == 4: return tensor  # global attention
    assert tensor.dim() == 5, str(tensor.dim())
    assert tensor.size(1) == 1, str(tensor.size(1))  # num_blocks
    seq_len = tensor.size(3)
    return tensor.squeeze(1)[:, :, :, -seq_len:]  # (bsz, num_blocks, H, seq_len, block_len) -> (bsz, H, seq_len, seq_len)

def append_tokens_to_positions(position_tensor):
    positions = numpy(position_tensor)
    return ['%d %s' % (p, tokens[p]) for p in positions]

def getdelattr(obj, name):
    r = getattr(obj, name, None)
    if hasattr(obj, name): delattr(obj, name)
    return r

def try_delattr(obj, name):
    if hasattr(obj, name): delattr(obj, name)

def get_attn_module(block):
    m = block.attn
    if hasattr(m, 'attention'): m = m.attention  # for gpt-neo
    return m


# +
def heatmap(a, figsize=(20, 1), cbar=False):
    _ = plt.figure(figsize=figsize)
    _ = sns.heatmap(numpy(a, decimals=None), cbar=cbar)
    plt.show()
    
def plot(a, figsize=(20, 2)):
    _ = plt.figure(figsize=figsize)
    _ = plt.plot(numpy(a))
    
def plot_hidden(hidden, topk=4):
    if hidden.dim() == 3 and hidden.size(0) == 1: hidden = hidden.squeeze(0)
    assert hidden.dim() == 2, str(hidden.dim())
    heatmap(hidden, figsize=(20, 5))
    hidden_mean = hidden.mean(dim=0)
    _ = plt.figure(figsize=(20, 2)); plt.xlim((0, hidden.size(1))); plt.plot(numpy(hidden_mean))
    return hidden_mean.topk(topk), hidden_mean.topk(topk, largest=False)

def plot_top_weight(weight, topk=4):
    wm = weight.norm(dim=-1)
    plot(wm, figsize=(20, 2))
    values, indices = wm.topk(topk)
    heatmap(weight[indices], figsize=(20, 1))
    return values, indices


# +
def unravel(i): return i // hidden_size, i % hidden_size
def indices_fn(indices): return [unravel(i) for i in numpy(indices)]

# wvo = wo.matmul(wv)
# show_topk(*wvo.view(-1).topk(5), indices_fn=indices_fn)
# show_topk(*wvo.view(-1).topk(5, largest=False), indices_fn=indices_fn)

def attn_out_transform(self, attn_out, alpha=1.0):
    wv = self.v_proj.weight.view(H, -1, hidden_size)[head]
    i = wv.norm(dim=0).argmax().item()
    w0, w1 = wv[:, i], attn_out[0, head, src]
    attn_out[0, head, src] = w0 * (w1.max() / w0.max() + w1.min() / w0.min()) / 2 * alpha
    return attn_out

def get_detach_fn(pos=None):
    def detach(hidden):
        if pos is None: return hidden.detach()
        h0, h1, h2 = hidden[:, :pos], hidden[:, pos: pos + 1], hidden[:, pos + 1:]
        h1 = h1.detach()
        return torch.cat([h0, h1, h2], dim=1)
    return detach

def get_detach_heads_fn(kept_head=None):
    def detach_heads(attn_weights):
        if kept_head is None: return attn_weights.detach()
        assert attn_weights.dim() == 4
        h0, h1, h2 = (
            attn_weights[:, :kept_head],
            attn_weights[:, kept_head : kept_head + 1],
            attn_weights[:, kept_head + 1 :],
        )
        h0, h2 = h0.detach(), h2.detach() 
        return torch.cat([h0, h1, h2], dim=1)
    return detach_heads

def get_scale_fn(factor=0):
    def scale(hidden): return hidden * factor
    return scale

def plot_attn(attn, annot=False, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    res = sns.heatmap(
        numpy(attn),
        square=True,
        cbar=False,
        annot=annot,
        fmt="d",
        linewidths=0.1,
        linecolor="grey",
        xticklabels=tokens,
        yticklabels=tokens,
    )
    _ = res.set_xticklabels(res.get_xmajorticklabels(), fontsize=8, rotation=0)
    _ = res.set_yticklabels(res.get_ymajorticklabels(), fontsize=8, rotation=0)
    # _ = plt.xlabel('%d-%d    %.4f' % (layer, head, v), fontsize=14)
    res.tick_params(top=True, right=True, labeltop=True, labelright=True)
    plt.show()

def cluster(emb, labels, n_clusters=3):
    assert emb.shape[0] == labels.shape[0], '%d ！= %d' % (emb.shape[0], labels.shape[0])
    centroids = emb.reshape(n_clusters, len(labels) // n_clusters, emb.shape[-1]).mean(axis=1)
    kmeans = KMeans(n_clusters=n_clusters)#, init=centroids)
    labels_ = kmeans.fit(emb).labels_
    for label in list(set(labels)):
        if Counter(labels_[labels == label]).most_common()[0][1] < (labels == label).sum():# - abs(label):
#             print(label)
            return False, labels_
    return True, labels_

def visualize_by_pca(emb, labels):
    pca = PCA(n_components=2)
    data = pca.fit_transform(emb)
    _ = plt.scatter(
        data[:, 0], data[:, 1], c=labels, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("jet", 3))
    _ = plt.colorbar()
    plt.show()

def get_query(self, h):
    query = self.q_proj(h)
    query = self._split_heads(query, self.num_heads, self.head_dim)
    query = query[0, head2, src:src+1]
    return query

def get_key(self, h):
    key = self.k_proj(h)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    key = key[0, head2, :]
    return key

def get_head_weights(layer, head):
    m = get_attn_module(blocks[layer])
    wq = m.q_proj.weight.view(H, -1, hidden_size)[head]
    wk = m.k_proj.weight.view(H, -1, hidden_size)[head]
    wv = m.v_proj.weight.view(H, -1, hidden_size)[head]
    wo = m.out_proj.weight.view(hidden_size, H, -1)[:, head]
#     return wq, wk, wv, wo
    return wq.t(), wk, wv.t(), wo.t()

def plot_tgt_attn(a, ax=None, title=None):
#     print(a.view(-1)[tgt_positions[4:]].mean())
    labels = np.array(tokens).reshape(nrows, -1)
    relative_tgt_positions = tgt_positions % a.size(1) # == ncols + 3
    right_attn = a.argmax(1) == relative_tgt_positions
    yticklabels = ['' if i else 'x' for i in right_attn]
    if ax is None:
        _ = plt.figure(figsize=(2.5 * a.size(1) / 9, 5 * a.size(0) / 24))
        _ = sns.heatmap(numpy(a) ,cbar=False, annot=labels, fmt='', xticklabels=False, yticklabels=yticklabels)
        if title is not None: plt.title(title)
    else:
        _ = sns.heatmap(numpy(a), cbar=False, annot=labels, fmt="", xticklabels=False, yticklabels=yticklabels, ax=ax)
        if title is not None: ax.set_title(title)
#     plt.show()


# +
def gen_detach_pairs(module, exit_module, detach_type='output'):
    assert detach_type in ['output', 'residual']
    pairs = []
    for block in blocks:
        if module in [block, get_attn_module(block)]: pairs += [(block, 'ffn_%s_transform' % detach_type)]
        elif block == exit_module: break
        elif pairs: pairs += [(block, 'attn_%s_transform' % detach_type), (block, 'ffn_%s_transform' % detach_type)]
    return pairs

def gen_detach_heads_tuples(module, exit_module, kept_layer, kept_head):
    tuples = None
    for i, block in enumerate(blocks):
        if module in [block, get_attn_module(block)]: tuples = []
        elif block == exit_module: break
        elif tuples is not None:
            tuples.append((get_attn_module(block), 'attn_weights_transform',
                          get_detach_heads_fn(kept_head=kept_head if i == kept_layer else None)))
    return tuples

def forward(module, names, values=None, exit_module=None, extra_tuples=None,
            detach_type=None, detach_pos=None, kept_layer=None, kept_head=None):
    if type(names) != list: names, values = [names], [values]
    if type(names) == list and type(values) != list: values = [values for _ in range(len(names))]
    for name, value in zip(names, values): setattr(module, name, value)
    if exit_module is not None: setattr(exit_module, 'exit', True)
    if extra_tuples is not None:
        for m, name, val in extra_tuples: setattr(m, name, val)
    if detach_type is not None:
        detach_pairs = gen_detach_pairs(module, exit_module, detach_type=detach_type)
        for m, name in detach_pairs: setattr(m, name, get_detach_fn(detach_pos))
    if kept_head is not None:
        detach_tuples = gen_detach_heads_tuples(module, exit_module, kept_layer=kept_layer, kept_head=kept_head)
        for m, name, fn in detach_tuples: setattr(m, name, fn)
    try: outputs = model(**inputs, output_attentions=True, output_hidden_states=exit_module is not None)
    finally:
        if values[0] is None: embs = [getattr(module, name) for name in names]
        for name in names: try_delattr(module, name)
        if exit_module is not None: try_delattr(exit_module, 'exit')
        if detach_type is not None:
            for m, name in detach_pairs: try_delattr(m, name)
        if kept_head is not None:
            for m, name, _ in detach_tuples: try_delattr(m, name)
        if extra_tuples is not None:
            for m, name, _ in extra_tuples: try_delattr(m, name)
    if values[0] is None and len(names) == 1: embs = embs[0]
    return embs if values[0] is None else outputs


# -

def test(hidden, query, key=None, logits=None, always_show=False):
    if logits is None:
        if key is None:
            key = self.k_proj(hidden)
            key = self._split_heads(key, self.num_heads, self.head_dim)[0, head2]
        logits = (query * key).sum(dim=-1)
    else:
        always_show = True
    cand_pos = torch.LongTensor(cand_positions).view(-1, n_candidates)
    is_extremal = [logits[p] == logits[cand_pos[i]].max() for i, p in enumerate(tgt_positions)]
    if always_show or sum(is_extremal[1:]) / len(tgt_positions[1:]) > 0.9:
        logits[0] = logits[1]
        plot(logits)
        _ = plt.xticks(range(len(logits)), tokens)
        for p, b in zip(tgt_positions, is_extremal): plt.axvline(x=p, color='gray' if b else 'r')
        plt.show()
        probs = logits[cand_positions].view(-1, n_candidates).softmax(-1)[cand_is_tgt]
        print(numpy(probs), '\n', probs.mean())
        return True
    return False 


# +
def create_mask(from_positions, to_positions, accum=False):
    mask = torch.zeros(1, seq_len, seq_len)
    for i in range(0, nrows):
        if not accum:
            mask[:, from_positions[i], to_positions[i]] = 1
        else:
            mask[:, from_positions[i], to_positions[:i]] = 1 / i if i > 0 else 0
    return mask

combined_weights = {}

def get_combined_w(layer, head, qk=False):
    if (layer, head, qk) in combined_weights: return combined_weights[(layer, head, qk)]
    wq, wk, wv, wo = get_head_weights(layer, head)
    w = torch.matmul(wq, wk) if qk else torch.matmul(wv, wo)
    combined_weights[(layer, head, qk)] = w
    return w


# -

def plot_tgt_attn_losses(labels, losses, losses1):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(20, 4))
    losses, losses1 = [int(l*100) for l in losses], [int(l*100) for l in losses1]
    rects1 = ax.bar(x - width/2, losses, width, label='loss')
    rects2 = ax.bar(x + width/2, losses1, width, label='loss1')
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(labels)
    _ = ax.legend()
    _ = ax.bar_label(rects1, padding=3)
    _ = ax.bar_label(rects2, padding=3)


inverse_fns = {
    identity.__name__: identity, lower.__name__: upper, upper.__name__: lower, 
    double.__name__: single, x10.__name__: d10,
    to_cardinal.__name__: to_digit, to_ordinal.__name__: to_digit}
inverse_fns.keys()

# +
nl_des = [(1, "One", "1st", "First"), (2, "Two", "2nd", "Second"),
          (3, "Three", "3rd", "Third"), (4, "Four", "4th", "Fourth"),
          (5, "Five", "5th", "Fifth"), (6, "Six", "6th", "Sixth"),
          (7, "Seven", "7th", "Seventh"), (8, "Eight", "8th", "Eighth"),
          (9, "Nine", "9th", "Ninth"), (10, "Ten", "10th", "Tenth"),
          (11, "Eleven", "11th", "Eleventh"), (12, "Twelve", "12th", "Twelfth"),
          (13, "Thirteen", "13th", "Thirteenth"), (14, "Fourteen", "14th", "Fourteenth"),
          (15, "Fifteen", "15th", "Fifteenth"), (16, "Sixteen", "16th", "Sixteenth"),
          (17, "Seventeen", "17th", "Seventeenth"), (18, "Eighteen", "18th", "Eighteenth"),
          (19, "Nineteen", "19th", "Nineteenth"), (20, "Twenty", "20th", "Twentieth"),
          (30, "Thirty", "30th", "Thirtieth"), (40, "Forty", "40th", "Fortieth"),
          (50, "Fifty", "50th", "Fiftieth"), (60, "Sixty", "60th", "Sixtieth"),
          (70, "Seventy", "70th", "Seventieth"), (80, "Eighty", "80th", "Eightieth"),
          (90, "Ninety", "90th", "Ninetieth"), (100, "One hundred", "100th", "Hundredth"),
          (1000, "One thousand", "1000th", "Thousandth")]
order_str_2_int = {i[3]: i[0] for i in nl_des}
order_int_2_str = {i[0]: i[3] for i in nl_des}

def order2int(order_str): return order_str_2_int.get(order_str, 2) - 1

def order2str(order_int): return order_int_2_str.get(order_int + 1, 'Second') # index begin from 0


# +
all_letters = upper_letters + lower_letters + [double(l) for l in upper_letters]
all_digits = list(chain.from_iterable([[fn(i) for i in digits] for fn in [identity, x10, double]]))

digit_fns = [identity, identity, to_cardinal, to_ordinal, double, x10]
upper_letter_fns = [identity, lower,]# double]
lower_letter_fns = [identity, upper,]# double]
vocabs = [(upper_letters, upper_letter_fns, to_rand_digit), 
          (lower_letters, lower_letter_fns, to_rand_digit), 
          (digits, digit_fns, to_rand_letter)]


# +
def make_query_str(instruction, query):
    if instruction is None and query is None: return ''
    s = '.'
    if instruction is not None: s = s + ' ' + instruction
    if query is not None:
        if type(query) in [int, bool, str]: query = [query]
        if type(query) == dict:
            s = s + " " + "{" + ",".join(
                [" replace %s with %s" % (str(k), str(v)) for k, v in query.items()]) + " }"
        elif type(query) in [list,]:
            s = s + ' ' + ' '.join([str(i) for i in query])
    return s

def make_example_str(example, with_instruction=False):
    instruction, l, query, ans = example
    if type(ans) not in [Sequence, list]: ans = [ans]
    ans = [str(i) for i in ans]
    return '%s -> %s' % (' '.join(l) + make_query_str(
        instruction if with_instruction else None, query), ' '.join(ans))

def sample_rand_len(vocab, k): return sample(vocab, k=randint(1, k))


# +
# def ith_element(l, query=None): return seq(l).slice(2, 3)
def ith_element(l, query=None): return seq(l).enumerate().filter(_[0] == order2int(query)).select(_[1])
def ith_group(l, query=None): return seq(l).group_by(_).select(_[1]).slice(1, 2).flatten() #.distinct()# davinci F w/ and wo dist
# def element_at_index(l, query): return seq(l).slice(query, query + 1) # davinci F
def element_at_index(l, query): return seq(l).enumerate().filter(_[0] == 1).select(_[1])
def replace(l, query): return seq(l).map(lambda x: query.get(x, x))
def replace_with_the_other(l, query): # davinci F
    query = {k: (set(l) - {k}).pop() for k in l}
    return replace(l, query)
def replace_all_with(l, query): return seq(l).map(lambda x: query)  # davinci F?!
def interleave_with(l, query): return seq(l).flat_map(lambda x: [x, query])  # davinci T!!
def unique_elements(l, query=None): return seq(l).distinct() # davinci F
def how_many_unique_elements(l, query=None): return seq(l).distinct().len()  # davinci F
def how_many(l, query): return seq(l).filter(_ == query).len() # davinci F
def select_same_as(l, query): return seq(l).filter(_ == query) # simpler version of how_many. davinci F
def select_same_number_as(l, query):
    return seq(l).group_by(_).select(_[1]).filter(lambda x: len(x) == len(query)).flatten()  # F
def includes(l, query): return seq(l).union(seq(query)).distinct().len() == seq(l).distinct().len() # davinci F
def is_included_by(l, query): return seq(l).difference(seq(query)).empty() # davinci F

tasks = [
    (ith_element, None, sample, lambda l, vocab, k: "Second"),
    (ith_element, None, sample, lambda l, vocab, k: order2str(randint(0, 1))),
    (
        ith_group,
        None,
        lambda vocab, k: seq(sample(vocab, k)).map(lambda x: [x] * randint(1, 3)).flatten().list(),
        None,
    ),
    (element_at_index, lambda: upper_letters, sample, lambda l, vocab, k: randint(0, min(2, len(l) - 1))),
    (replace, None, sample, lambda l, vocab, k: {choice(l): choice(vocab)}),
    (
        replace_with_the_other,
        lambda: sample(upper_letters, 2),
        lambda vocab, k: sample(vocab + choices(vocab, k=k - 2), k),
        None,
    ),
    (replace_all_with, None, sample_rand_len, lambda l, vocab, k: choice(vocab)),
    (interleave_with, None, sample_rand_len, lambda l, vocab, k: choice(vocab)),
    (unique_elements, lambda: sample(upper_letters, 3), choices, None),
    (how_many_unique_elements, lambda: sample(upper_letters, 3), choices, None),
    (how_many, lambda: sample(upper_letters, 3), choices, lambda l, vocab, k: choice(list(set(l)))),
    (select_same_as, lambda: sample(upper_letters, 3), choices, lambda l, vocab, k: choice(list(set(l)))),
    (
        select_same_number_as,
        None,
        lambda vocab, k: seq(sample(vocab, k)).map(lambda x: [x] * randint(1, 3)).flatten().list(),
        lambda l, vocab, k: [choice(vocab)] * randint(1, 3),
    ),
    (includes, lambda: sample(upper_letters, 6), sample, lambda l, vocab, k: sample(vocab, 3)),
    (is_included_by, lambda: sample(upper_letters, 6), sample, lambda l, vocab, k: sample(vocab, 5)),
]

# +
full_vocab = string.ascii_uppercase + string.digits
transform_fn, vocab_fn, sample_fn, query_fn = tasks[1]
instruction = transform_fn.__name__.replace('_', ' ')
if vocab_fn is None: vocab_fn = lambda: full_vocab
if query_fn is None: query_fn = lambda *_: None
nrows, ncols = 16, 4
examples = []
query = None
for i in range(nrows):
    vocab = vocab_fn()
    l = sample_fn(vocab, k=ncols)
    query = query_fn(l, vocab, ncols)
    examples.append([instruction, l, query, transform_fn(l, query=query)])
examples

text = '\n'.join([make_example_str(e, with_instruction=False) for e in examples])
text = '\n' + text + '\n'
print(text)
# -

# task_name = 'find majority'  ##?
# task_name = 'find special kind'  ****
# task_name = 'is same' / 'is same kind'  ****
# task_name = 'find special easy' ## 6-2
# task_name = 'A B C -> B' ##
# task_name = 'set diff' ##?
# task_name = 'A(BC->B' ##  6-1
# task_name = 'ABC,AXC->X' ##?
# task_name = 'reverse set diff' ##, *failed only on first position, GPT-3 has this problem, too
# task_name = 'reverse set diff v2' ## A*C,ABC->B
# task_name = 'find next easy' ## ABCDEF,\nBC->D, 6+2
inputs = tokenizer.encode_plus(text, return_tensors='pt')
inputs = prepare_inputs(inputs, model.device)
with torch.no_grad(): outputs = model(**inputs, output_attentions=True)
input_ids = inputs.input_ids
logits = outputs.logits
bos_id = tokenizer._convert_token_to_id('Ġ->')
crlf_id = tokenizer._convert_token_to_id('Ċ')
bsz = input_ids.size(0); assert bsz == 1
labels = torch.ones_like(input_ids) * (-100)
for bi in range(bsz):
    bos_indices = (input_ids[bi] == bos_id).nonzero().squeeze(1)
    eos_indices = (input_ids[bi] == crlf_id).nonzero()[-nrows:].squeeze(1)
    for i, (example, bos_i, eos_i) in enumerate(zip(examples, bos_indices.tolist(), eos_indices.tolist())):
        print(i, end='\t')
        print(' ' + make_example_str(example), end='\t')
        ans_ids = input_ids[bi, bos_i + 1: eos_i]
        if i >= 2: labels[bi, bos_i: eos_i - 1] = ans_ids
        ans_prob_dist = logits[bi, bos_i: eos_i - 1].softmax(-1)
        ans_probs = ans_prob_dist[torch.arange(ans_prob_dist.size(0)), ans_ids]
        # for bi sample in every batch, fetch answer prob
        ans_tokens = tokenizer.convert_ids_to_tokens(ans_ids)
        for ans_id, ans_token, ans_prob, dist in zip(ans_ids, ans_tokens, numpy(ans_probs, decimals=3), ans_prob_dist):
            top1_correct = (dist.argmax() == ans_id).item()
            print(('*' if top1_correct else ' ') + ans_token, ans_prob, 
                  show_topk(*dist.topk(5), indices_fn=tokenizer.convert_ids_to_tokens)) 
loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
loss

# +
bos_positions = (input_ids == bos_id).nonzero()[:,1]
# the position before ans
ans_positions = bos_positions + 1

bos_watch_ind = 1

src = bos_positions[1].item()
# select answer position
pred_label = outputs.logits[0, src].argmax().item()
# predicted char on final answer position
tokens = [token.replace('Ġ', '').replace('Ċ', '^') for token in tokenizer.tokenize(text)]
seq_len = len(tokens)
answer = tokens[src + 1]
# standard final answer
cand_range = range(eos_indices[bos_watch_ind - 1] + 1, bos_indices[bos_watch_ind])
# condidates chars appreared in current sample (same line)
n_candidates = len(cand_range); assert n_candidates >= 1, str(n_candidates)
ans_fn = lambda x: x
tgt = [i for i in cand_range if ans_fn(tokens[i]) == answer][0] if n_candidates > 1 else cand_range[0]
# cand_positions = [i for i, token in enumerate(tokens[:-1]) if '^' in tokens[max(0, i - n_candidaes): i]]
# -


start_positions = (input_ids[bi] == crlf_id).nonzero()[:-1, 0]
dot_id = tokenizer._convert_token_to_id('.')
dot_positions = (input_ids[bi] == dot_id).nonzero()[:, 0]
ans_fn = lambda x: x

tgt_positions = []
for i in range(len(start_positions)):
    start_pos, end_pos, ans_pos = start_positions[i], dot_positions[i], ans_positions[i]
    for pos in range(start_pos, end_pos):
        if ans_fn(tokens[pos]) == tokens[ans_pos]: tgt_positions.append(pos)
tgt_positions = torch.LongTensor(tgt_positions)
assert len(tgt_positions) == len(ans_positions), '%d != %d' % (len(tgt_positions), len(ans_positions))
# cand_is_tgt = torch.LongTensor(cand_positions).view(-1, n_candidates) == tgt_positions.unsqueeze(-1)

# +
for i, block in enumerate(blocks[:]):
    block.attn_output, block.ffn_output = None, None
    am = get_attn_module(block)
    am.attention_mask, am.head_output, am.attn_out = None, None, None
# get_attn_module(blocks[10]).hidden_states_mask = h_mask  ######
# get_attn_module(blocks[layer2]).return_attn_logits = True
try: 
    with torch.no_grad(): outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
finally:
    attn_outputs, ffn_outputs, attn_hidden_states, attention_masks, head_outputs, attn_outs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, block in enumerate(blocks[:]):
        attn_outputs.append(getdelattr(block, 'attn_output'))
        ffn_outputs.append(getdelattr(block, 'ffn_output'))
        am = get_attn_module(block)
        attention_masks.append(getdelattr(am, 'attention_mask'))
        head_outputs.append(getdelattr(am, 'head_output'))
        attn_outs.append(getdelattr(am, 'attn_out'))

#     try_delattr(get_attn_module(blocks[10]), 'hidden_states_mask')   ######
#     try_delattr(get_attn_module(blocks[layer2]), 'return_attn_logits')
hidden_states = outputs.hidden_states
attentions = outputs.attentions

outputs.attn_outputs = attn_outputs
outputs.ffn_outputs = ffn_outputs
outputs.attn_outs = attn_outs
outputs.head_outputs = head_outputs
# -

layer_out = L
for i in range(0, L - 2 + 2):
    h = hidden_states[i]
    if i < L - 1: h = blocks[-2](h, attention_mask=attention_masks[-2])[0]
    h = blocks[-1](h, attention_mask=attention_masks[-1])[0]
    h = model.transformer.ln_f(h)
    h = h[0, src]
    logits = model.lm_head(h)
    print(i, show_topk(*logits.softmax(-1).topk(5), indices_fn=tokenizer.convert_ids_to_tokens))
    if logits.argmax() == input_ids[0, src + 1] and layer_out == L: layer_out = i
print(layer_out)

attr_names = ['attn_output', 'ffn_output', 'attention_mask', 'attn_out']#, 'head_out']
for i, block in enumerate(blocks):
    for name in attr_names:
        m = block if name.endswith('output') else get_attn_module(block)
        setattr(m, name, None)
try: 
    with torch.no_grad(): o = model(**inputs, output_attentions=True, output_hidden_states=True)
finally:
    for i, block in enumerate(blocks):
        for name in attr_names:
            m = block if name.endswith('output') else get_attn_module(block)
            if not hasattr(o, name): setattr(o, name, [])
            getattr(o, name).append(getdelattr(m, name))
hidden_states = o.hidden_states
attentions = o.aw = o.attentions

# +
task_name = 'find majority'  ##?


# grad attribution
pred_attn = False
keys = ['aw'] #  'attn_out', 'head_out', 'attn_output', 'ffn_output'
keys2 = [] #['head_output', 'attn_output']
layer_range = (0, layer1) if pred_attn else (0, layer_out)
# layer_out: total layers
attrs, grads = defaultdict(list), defaultdict(list)
num_points, batch_size = 5, 5

layer0 = 9
    
for i in tqdm(range(*layer_range)):
    am = get_attn_module(blocks[i])

    scaled_emb, step, grad = {}, {}, {}
    embs = [getattr(o, keys[0])[i]]
    # model.attentions all attentions, [i], layer_i
    if len(embs) == 1 and keys[0] != 'aw': all_embs[task_name][keys[0]] = embs[0][0]
    # embs[0] batch_ind 0, embs[0][0], batch_ind 0, head 0
        
    for key, emb in zip(keys, embs):
        scaled_emb[key], step[key] = scaled_input(emb, num_points)
        _ = scaled_emb[key].requires_grad_(True)
        grad[key] = None
    if i == layer0: ys = []
    # why layer 9 need deal with special
    for j in range(0, num_points, batch_size):
        sliced_scaled_emb = [scaled_emb[key][j: j + batch_size] for key in keys]
        outputs = forward(am, keys, values=sliced_scaled_emb, exit_module=blocks[layer1+1] if pred_attn else None)
        # tocreate: exit_module ? only forward one layers?
        y = (
            globalize(outputs.attentions[layer2])[:, head2, src, tgt]
            if pred_attn
            else outputs.logits.softmax(-1)[:, src, pred_label]
        )
        # after change attention, get src predicted result
        if i == layer0: ys.append(y);
#         if keys2:
#             sliced_scaled_emb2 = [getdelattr(am if key in ['head_output'] else blocks[i], key) for key in keys2]
#             sliced_scaled_emb += sliced_scaled_emb2
#             if j == num_points - batch_size: step.update({key: emb[-1:]/num_points for key, emb in zip(keys2, sliced_scaled_emb2)})
        sliced_grads = torch.autograd.grad(y.flatten().unbind(), sliced_scaled_emb)
        for gi, key in enumerate(keys + keys2):
            # you wrap key and grad in dict format at forward, so extract it here
            sliced_grad = sliced_grads[gi].sum(dim=0, keepdim=True)
            # sum across head
            grad[key] = sliced_grad if key not in grad or grad[key] is None else grad[key] + sliced_grad
    for key in keys + keys2:
        attr = grad[key] * step[key]
        attrs[key].append(attr.data)
        grads[key].append(grad[key].data)


if len(keys) == 1:
    key = keys[0]
    all_attrs[task_name][key + str(int(pred_attn))] = torch.cat([globalize(a) for a in attrs[key]]) \
        if key == 'aw' else attrs[key][0][0]
#     for key in keys2: attrs[key] = torch.cat(attrs[key])
# -

for i, token in enumerate(tokens):
    if token in ['Ċ', '^']: print()
    else: print('%4d %-6s' %(i, token), end='  ')
tgt_positions


def show_top_heads(values, indices, src_indices=None, tgt_indices=None, topk=15):
    val, ind = values.sum(dim=-1).view(-1).topk(topk)
    # sort by importance across layer and head
    val, ind = numpy(val), unravel_index(ind, values.size()[:-1])
    # get topk head importance and it's index
    for (l, h), v in zip(ind, val):
        _l = l + layer_range[0]
        if _l <= 3: continue
        top_links = list(zip(unravel_index(indices[l, h], (seq_len, seq_len)), numpy(values[l, h], decimals=3)))
        # for each head, which position is important, and deserve to attend
        if src_indices is not None: top_links = [([src_indices[_s], _t], _v) for [_s, _t], _v in top_links]
        if tgt_indices is not None: top_links = [([_s, tgt_indices[_t]], _v) for [_s, _t], _v in top_links]
        top_links = [
            ([_s, _t], _v, numpy(globalize(attentions[_l]) * 100, decimals=1)[0, h, _s, _t])
            for [_s, _t], _v in top_links
        ]
        _top_links = [([_s, _t], _v, _a) if len(src_indices) > 1 else (_t, _v, _a) for [_s, _t], _v, _a in top_links]
        print('%d-%d\t%.3f' % (_l, h, v), _top_links, end='\t') 
        if len(top_links) == 1:
            probs = numpy(globalize(attentions[_l])[0, h, src])
            for i in cand_range:
                # consider the candidate position
                if i == tgt: print('*', end='')
                # if only concern the total attention, whether current head attend to tgt position
                print('%.10f' % probs[i], end='  ')
        print()


a = all_attrs[task_name]['aw' + str(int(pred_attn))]
# layers, heads, src, tgt aggreation importance
a = a / a.view(a.size(0), -1).norm(dim=1)[:, None, None, None] #.view(a.size(0), 1, 1, 1)
# why view as (layer, None) without consider head, and then norm

if not pred_attn:
    src_indices, tgt_indices = [src], [tgt]
    _a = a[:, :, src_indices, tgt_indices]
    values, indices = _a.view(_a.size(0), H, -1).topk(1, dim=-1)
    # value is importance, and indices is direct from src_indices to tgt_indices, in this case always 0 -> 0
    # cause these two list have only one element
    show_top_heads(values, indices, src_indices=src_indices, tgt_indices=tgt_indices)
    # use integrate gradient method, which head is most important
    # layer-head, head-importance, (tgt_attention, head_importance, attention_score * 100), '\t'.join(candidate's scores)
    print()
# src_indices = numpy(ans_positions[:])
# src_indices = numpy(tgt_positions + 1)
# tgt_indices = tgt_positions
_a = a[:, :, src_indices, :]
values, indices = _a.view(_a.size(0), H, -1).topk(nrows // 2, dim=-1)
show_top_heads(values, indices, src_indices=src_indices)#, tgt_indices=tgt_indices)
