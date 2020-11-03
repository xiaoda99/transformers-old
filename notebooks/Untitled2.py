#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[3]:


import sys
sys.path.insert(0, '/nas/xd/projects/transformers/src/transformers')
import os
device_mappings = {0: 1, 1: 5, 2: 6, 3: 7, 4: 2, 5: 3, 6: 0, 7: 4}
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_mappings[5])

import random
import string
from collections import defaultdict
from itertools import product, chain
import numpy as np
from pattern.en import comparative

import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from enum import Enum
from typing import List, Optional, Union

from child_frames import frames
from utils import *

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers.modeling_roberta import RobertaDoubleHeadsModel  # XD

logging.basicConfig(level=logging.ERROR)


# In[5]:


A_template = "{rel_prefix} {dt} {ent0} {rel} {dt} {ent1} {rel_suffix}"
B_templates = ["{pred_prefix} {dt} {ent} {pred}", "{pred_prefix} {pred} {dt} {ent}"]
B_template = B_templates[0]
entailment_templates = [
    "{A} ? {conj} , {B} .",  # yes/no/maybe
    "{A} , so {B} ? {conj} .",
]
marker = '*'
def get_comparative(word, add_marker=True):
    compar = comparative(word)
    if add_marker:
        compar = compar.replace('more ', 'more %s ' % marker) if compar.startswith('more ') else marker + ' ' + compar
    return compar
    
def negate_sent(sent):
    assert ' is ' in sent
    neg_sents = []
    neg_sent0 = sent.replace(' is ', ' is not ') if 'more ' not in sent else sent.replace('more ', 'less ')
    neg_sents.append(neg_sent0)
    neg_sents.append('it is unlikely that ' + sent)
    return neg_sents

def strip_rel_id(s, lexical_rel=''):
    rel_id_span = s[s.index(':'): s.index(':') + 2]
    if lexical_rel != '': lexical_rel = ' ( ' + lexical_rel + ' )'
    return s.replace(rel_id_span, lexical_rel)
        
tag2id = {'same': 0, 'opposite': 1, 'unrelated': 2, 'former': 3, 'latter': 4, 'another': 5, 'single': 6, 'paired': 7}
tag2id = {'Ġ' + k: v for k, v in tag2id.items()}
id2tag = {v: k for k, v in tag2id.items()}

def make_sentences(index=-1, entities=["_X", "_Z"], entity_set=string.ascii_uppercase, determiner="",
                   relation_group=[["big",], ["small"]], rand_relation_group=[["short"], ["tall", "high"]],
                   relation_prefix="", relation_suffix="", predicate_prefix="",
                   n_entity_trials=3, has_negA=True, has_negB=True, has_neutral=False, mask_types={'sent_rel'}, 
                   lexical_relations=['same', 'opposite', 'unrelated'], tag_lexical_rel=False, tag_entity_rel=False):
#     if tag_lexical_rel: mask_types.add('lexical_rel')
#     if tag_entity_rel: mask_types.add('entity_rel')
        
    def form_As(relations):
        return [A_template.format(dt=determiner, ent0=ent0, ent1=ent1, rel=rel, rel_prefix=relation_prefix, rel_suffix=relation_suffix)
              for ent0, ent1, rel in [entities + relations[:1], reverse(entities) + reverse(relations)[:1]]]

    As = []
    for rel0 in relation_group[0]:
        for rel1 in relation_group[1]:
            relations = ["is %s:%d than" % (get_comparative(rel), i) for i, rel in enumerate([rel0, rel1])]
            As += form_As(relations)
    As = list(set(As))
    negAs = join_lists([negate_sent(A)[:1] for A in As]) if has_negA else []

    def form_Bs(predicates): 
        f = mask if 'entity' in mask_types else (lambda x: x)
        return [B_template.format(dt=determiner, ent=f(ent), pred=pred, pred_prefix=predicate_prefix)
              for ent, pred in zip(entities, predicates)]

    Bs, negBs = {'orig': [], 'rand': []}, {}
    for k, group in zip(['orig', 'rand'], [relation_group, rand_relation_group]):
        for rel0 in group[0]:
            for rel1 in group[1]:
                predicates = ["is %s:%d" % (get_comparative(rel), i) for i, rel in enumerate([rel0, rel1])]
                Bs[k] += form_Bs(predicates)
    for k in Bs:
        Bs[k] = list(set(Bs[k]))
        if has_negB:
            negBs[k] = join_lists([negate_sent(B)[:1] for B in Bs[k]])
            Bs[k], negBs[k] = Bs[k] + [swap_entities(negB) for negB in negBs[k]], negBs[k] + [swap_entities(B) for B in Bs[k]]
        else:
            negBs[k] = [swap_entities(B) for B in Bs[k]]

    def form_sentences(sentence_template, As, Bs, conj):
        def extract_rel_id(s): return int(s[s.index(':') + 1])
        def get_lexical_rel(rel_id_A, rel_id_B):
            return 'same' if rel_id_A == rel_id_B else 'opposite'
        def tag_token(token, recurred_entity):
            if token not in entities: return token
            entity_rel = 'paired' if token == recurred_entity else 'single'
            if 'entity_rel' in mask_types: entity_rel = mask(entity_rel)
            return token + ' ( %s )' % entity_rel
        def compare_and_tag_entity(A, B):
            recurred_entity = [e for e in entities if e in B.split()][0]
#             entity_rel = 'former' if A.strip().startswith(recurred_entity) else 'latter'
#             if 'entity_rel' in mask_types: entity_rel = mask(entity_rel)
#             return B.replace(entity, entity + ' ( ' + entity_rel + ' )')
            A = ' '.join([tag_token(token, recurred_entity) for token in A.split()])
            B = ' '.join([tag_token(token, recurred_entity) for token in B.split()])
            return A, B                 
        
        if 'sent_rel' in mask_types: conj = mask(conj)
        As_with_rel_ids = [(A, extract_rel_id(A)) for A in As]
        Bs_with_rel_ids = [(B, extract_rel_id(B)) for B in Bs]
            
        sentences = []
        for (A, rel_id_A), (B, rel_id_B) in product(As_with_rel_ids, Bs_with_rel_ids):
            lexical_rel = 'unrelated' if 'Maybe' in conj                 else get_lexical_rel(rel_id_A, rel_id_B)
            if lexical_rel in lexical_relations:
                if tag_entity_rel: A, B = compare_and_tag_entity(A, B)
                if not tag_lexical_rel: lexical_rel = ''
                elif 'lexical_rel' in mask_types: lexical_rel = mask(lexical_rel)
                sent = sentence_template.format(A=strip_rel_id(A), 
                                                B=strip_rel_id(B, lexical_rel), 
                                                conj=conj)
                sent = " " + " ".join(sent.split())
                sentences.append(sent)
        return sentences

    sentences = defaultdict(list)
    for entailment_template in entailment_templates[-1:]:
        for A, B, conj in [(As, Bs['orig'], 'Right'), 
                           (negAs, negBs['orig'], 'Right'), 
                           (As, negBs['orig'], 'Wrong'), 
                           (negAs, Bs['orig'], 'Wrong'),
                           (As, Bs['rand'], 'Maybe'), 
                           (negAs, negBs['rand'], 'Maybe'), 
                           (As, negBs['rand'], 'Maybe'), 
                           (negAs, Bs['rand'], 'Maybe'),
                          ]:
            sentences[conj] += form_sentences(entailment_template, A, B, conj)
    assert len(sentences['Right']) == len(sentences['Wrong']),         '%d %d' % (len(sentences['Right']), len(sentences['Wrong']))
    if has_neutral:
        sentences['Maybe'] = random.sample(sentences['Maybe'], len(sentences['Right']))
    keys = sentences.keys() if has_neutral else ['Right', 'Wrong']
    sentences = join_lists(sentences[k] for k in keys)
    
    substituted_sent_groups = []
    for sent in sentences:
        sent_group = []
        for _ in range(n_entity_trials):
            e0, e1 = random.sample(entity_set, 2)
            sent_group.append(sent.replace(entities[0], e0).replace(entities[1], e1))
        substituted_sent_groups.append(sent_group)
    return sentences, substituted_sent_groups

make_sentences(has_negA=True, has_negB=True, has_neutral=False, tag_lexical_rel=False, tag_entity_rel=True, 
               mask_types={'sent_rel', 'entity_rel'})[0]


# In[6]:





# In[7]:


model_class, tokenizer_class, shortcut = RobertaForMaskedLM, RobertaTokenizer, 'roberta-large'
# model_class = RobertaDoubleHeadsModel
model, tokenizer = None, tokenizer_class.from_pretrained(shortcut)


# In[8]:


random.shuffle(frames)
n_entity_trials = 10
all_lines = [make_sentences(relation_group=rg, rand_relation_group=frames[(i + 1) % len(frames)], 
                            n_entity_trials=n_entity_trials, 
                            has_negA=True, has_negB=True, tag_lexical_rel=False, tag_entity_rel=True,
                            has_neutral=False, mask_types={'sent_rel', 'entity_rel'})[1] 
             for i, rg in enumerate(frames)]
# all_lines = [make_transitive(relation_group=rg, n_entity_trials=10, 
#                              has_negP=False, has_negQ=False, has_neutral=False, mask_types=['sent_rel'])[1] 
#              for i, rg in enumerate(frames)]
# all_lines = join_lists(all_lines)
# all_lines = join_lists(all_lines)
tokenizer.tag2id, tokenizer.id2tag = tag2id, id2tag
for k in CHILDDataset.all_lines: CHILDDataset.all_lines[k] = None
train_dataset = CHILDDataset(all_lines, tokenizer, has_markers=True, has_tags=False, max_noise_len=0, split_pct=[0.7, 0.3, 0.0], mode='train')
eval_dataset = CHILDDataset(all_lines, tokenizer, has_markers=True, has_tags=False, max_noise_len=0, split_pct=[0.7, 0.3, 0.0], mode='dev')
print('nTrain = %d, nValid = %d' % (len(train_dataset), len(eval_dataset)))


# In[9]:


# model = RobertaDoubleHeadsModel.from_pretrained('roberta-base', model=model)
model = model_class.from_pretrained('roberta-base', model=model)
steps = int(round(100 * n_entity_trials / 3))
training_args = TrainingArguments(output_dir="./models/model_name", 
    overwrite_output_dir=True, do_train=True, do_eval=True,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    learning_rate=2e-5, num_train_epochs=3,
    logging_steps=steps, eval_steps=steps, save_steps=0,
    no_cuda=False, evaluate_during_training=True,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.tokenizer = tokenizer


# In[ ]:


trainer.train()

"""
# In[102]:


trainer.evaluate()


# In[80]:


model.roberta.selectively_masked_head = (5, 5)


# In[78]:


trainer.args.per_device_eval_batch_size


# In[10]:


dataloader = DataLoader(
    eval_dataset,
    sampler=RandomSampler(eval_dataset),
    batch_size=trainer.args.eval_batch_size,
    collate_fn=trainer.data_collator,
    drop_last=trainer.args.dataloader_drop_last,
)


# In[11]:


for inputs in dataloader: break

for i in range(20):
    print(i, tokenizer.decode(inputs['input_ids'][i]), 
          [tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])[j] for j in inputs['marked_positions'][i]])


# In[19]:


_ = model.eval()
inputs = trainer._prepare_inputs(inputs, model)
with torch.no_grad():
    loss, logits, attns = model(**inputs, output_attentions=True)

attn_scores, attn_probs = zip(*attns)
attn_scores, attn_probs = torch.stack(attn_scores, dim=0), torch.stack(attn_probs, dim=0)
attn_probs = attn_probs.cpu()


# In[12]:


def normalize_tokens(tokens):
    return ['@' + token if not token.startswith('Ġ') and token not in ['<s>', '</s>', '<mask>'] else token.replace('Ġ', '') 
                  for token in tokens] 


# In[20]:


sample_indices = [[2, 6, 7, 11], 
                  [1, 3, 4, 5]]
n_rows, n_cols = len(sample_indices), len(sample_indices[0])
fig, axs = plt.subplots(n_rows, n_cols, sharey=False, figsize=(4 * n_cols, 4.5 * n_rows))
for row in range(n_rows):
    for col in range(n_cols):
        i, ax = sample_indices[row][col], axs[row][col]
        p, h = inputs['marked_positions'][i]
        p, h = p.item(), h.item()
        pos_attn = attn_probs[:, i, :, h, p]
        tokens = normalize_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
        ax = sns.heatmap(pos_attn, square=True, cbar=False, ax=ax)
        ax.tick_params(top=True, labeltop=True)
        _ = ax.set_xlabel('%s - %s' % (tokens[p], tokens[h]))


# In[14]:


sample_indices = [[2, 6, 7, 11], 
                  [1, 3, 4, 5]]
n_rows, n_cols = len(sample_indices), len(sample_indices[0])
fig, axs = plt.subplots(n_rows, n_cols, sharey=False, figsize=(4 * n_cols, 4.5 * n_rows))
for row in range(n_rows):
    for col in range(n_cols):
        i, ax = sample_indices[row][col], axs[row][col]
        p, h = inputs['marked_positions'][i]
        p, h = p.item(), h.item()
        pos_attn = attn_probs[:, i, :, h, p]
        tokens = normalize_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
        ax = sns.heatmap(pos_attn, square=True, cbar=False, ax=ax)
        ax.tick_params(top=True, labeltop=True)
        _ = ax.set_xlabel('%s - %s' % (tokens[p], tokens[h]))


# In[175]:


i = 4
layer, head = 5, 5
seq_len = inputs['attention_mask'][4].sum().item()
attn =  attn_probs[layer, i, head, : seq_len, : seq_len]
tokens = normalize_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])[: seq_len])
size = round(attn.size(0) / 3)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(size  * 2., size), gridspec_kw={'width_ratios': [3, 1]})
_ = sns.heatmap((attn * 100).long(), square=True, cbar=True, annot=False, fmt='d', xticklabels=tokens, yticklabels=tokens, ax=ax0)
plot_head_attn(attn, tokens, ax1=ax1, marked_positions=inputs['marked_positions'][i]) 


# In[1]:


from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
"""
