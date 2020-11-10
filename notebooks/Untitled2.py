#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


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

from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer


# In[3]:


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from enum import Enum
from typing import List, Optional, Union

from child_frames import frames
from utils import *

import logging
import os
import sys

from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers import PreTrainedModel, RobertaForMaskedLM, RobertaTokenizer
from transformers.modeling_roberta import RobertaDoubleHeadsModel, RobertaDoubleHeadsModel2, RobertaDoubleHeadsModel3  # XD

logging.basicConfig(level=logging.ERROR)


# In[4]:


loss_fct = CrossEntropyLoss()


# In[5]:


a = torch.arange(3 * 2 * 2).view(3, 2, 2)
b = torch.Tensor([True, False, True])
a[b.nonzero().squeeze(1)].size()


# In[18]:


A_template = "{rel_prefix} {dt} {ent0} {rel} {dt} {ent1} {rel_suffix}"
B_template = "{pred_prefix} {dt} {ent} {pred}"
entailment_templates = ["{A} ? {conj} , {B} .", "{A} , so {B} ? {conj} ."]

markers = {'lexical': '*', 'entity': '#'}

def extract_rel_id(s): return int(s[s.index(':') + 1])

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
                   lexical_relations=None, tag_lexical_rel=False, tag_entity_rel=False):
    if tag_lexical_rel: mask_types.add('lexical_rel')
    if tag_entity_rel: mask_types.add('entity_rel')  
    def get_comparative(word):
        compar = comparative(word)
        if tag_lexical_rel:
            marker = markers['lexical']
            compar = compar.replace('more ', 'more %s ' % marker) if compar.startswith('more ') else marker + ' ' + compar
        return compar
  
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
        def compare_and_tag_comparative(A, B):
            if 'Maybe' in conj:
                lexical_rel = 'unrelated'
            else:
                lexical_rel = 'same' if extract_rel_id(A) == extract_rel_id(B) else 'opposite'
            if lexical_rel in ['same', 'opposite', 'unrelated'] and lexical_relations and lexical_rel not in lexical_relations:
                return None, None
            if 'lexical_rel' in mask_types: lexical_rel = mask(lexical_rel)
            return strip_rel_id(A), strip_rel_id(B, lexical_rel)
        def tag_entity(token, recurred_entity, entity_rel=None):
            if token != recurred_entity: return token
            return markers['entity'] + ' ' + token
#             if token not in entities: return token
#             entity_rel = 'paired' if token == recurred_entity else 'single'
#             if 'entity_rel' in mask_types: entity_rel = mask(entity_rel)
#             return token + ' ( %s )' % entity_rel
            
        def compare_and_tag_entity(A, B):
            recurred_entity = [e for e in entities if e in B.split()][0]
            A = ' '.join([tag_entity(token, recurred_entity) for token in A.split()])
            B = ' '.join([tag_entity(token, recurred_entity) for token in B.split()])
            entity_rel = 'former' if A.split().index(recurred_entity) in [0, 1] else 'latter'
            if 'entity_rel' in mask_types: entity_rel = mask(entity_rel)
            return A, B.replace(recurred_entity, recurred_entity + ' ( ' + entity_rel + ' )')
        
        if 'sent_rel' in mask_types: conj = mask(conj)
        sentences = []
        for A, B in product(As, Bs):
            A, B = compare_and_tag_comparative(A, B) if tag_lexical_rel else (strip_rel_id(A), strip_rel_id(B))
            if A is None: continue
            if tag_entity_rel: A, B = compare_and_tag_entity(A, B)
            sent = sentence_template.format(A=A, B=B, conj=conj)
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
    n0 = sum('same' in sent for sent in sentences)
    n1 = sum('opposite' in sent for sent in sentences)
#     assert n0 == n1, '%d != %d %s' % (n0, n1, str(relation_group))
    
    substituted_sent_groups = []
    for sent in sentences:
        sent_group = []
        for _ in range(n_entity_trials):
            e0, e1 = random.sample(entity_set, 2)
            sent_group.append(sent.replace(entities[0], e0).replace(entities[1], e1))
        substituted_sent_groups.append(sent_group)
    return sentences, substituted_sent_groups

make_sentences(has_negA=True, has_negB=True, has_neutral=False, tag_lexical_rel=True, tag_entity_rel=True,
               mask_types={'sent_rel', 'lexical_rel', 'entity_rel'})[0]


# In[7]:


tokenizer_class, shortcut = RobertaTokenizer, 'roberta-base'
tokenizer = tokenizer_class.from_pretrained(shortcut)


# In[24]:


random.seed(42)
random.shuffle(frames)
n_entity_trials = 10
all_lines = [make_sentences(relation_group=rg, rand_relation_group=frames[(i + 1) % len(frames)], 
                            n_entity_trials=n_entity_trials, 
                            has_negA=True, has_negB=True, tag_lexical_rel=True, tag_entity_rel=True,
                            has_neutral=False, mask_types={'sent_rel',})[1] # 'entity_rel', 'lexical_rel'
             for i, rg in enumerate(frames)]
# all_lines = join_lists(all_lines)
# all_lines = join_lists(all_lines)
tokenizer.tag2id, tokenizer.id2tag = tag2id, id2tag
for k in CHILDDataset.all_lines: CHILDDataset.all_lines[k] = None
train_dataset = CHILDDataset(all_lines, tokenizer, markers=markers, has_tags=False, split_pct=[0.7, 0.3, 0.0], mode='train')
eval_dataset = CHILDDataset(all_lines, tokenizer, markers=markers, has_tags=False, split_pct=[0.7, 0.3, 0.0], mode='dev')
print('nTrain = %d, nValid = %d' % (len(train_dataset), len(eval_dataset)))


# In[ ]:


# model_class, model = RobertaForMaskedLM, None
model_class, model = RobertaDoubleHeadsModel3, None

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
model.tokenizer = tokenizer
trainer.train()

"""
# In[182]:


output_dir = '/nas/xd/data/models/CHILD/pred_all_rels/'
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)


# In[80]:


model.roberta.selectively_masked_head = (5, 5)


# In[91]:


dataloader = DataLoader(
    eval_dataset,
    sampler=RandomSampler(eval_dataset),
    batch_size=trainer.args.eval_batch_size,
    collate_fn=trainer.data_collator,
    drop_last=trainer.args.dataloader_drop_last,
)


# In[92]:


for inputs in dataloader: break

for i in range(20):
    print(i, tokenizer.decode(inputs['input_ids'][i])) #, 
#           [tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])[j] for j in inputs['head_mask'][i]])


# In[140]:


marked_pos_labels = inputs['marked_pos_labels']
marked_pos_labels.size()


# In[128]:


holoattn = torch.cat(attn_probs, dim=1)
holoattn.size()


# In[129]:


marked_holoattn = holoattn[torch.arange(64).unsqueeze(-1), :, marked_pos_labels[:, :, 0], marked_pos_labels[:, :, 1]]
marked_holoattn.size()


# In[131]:


marked_hidden = last_hidden_states[((inputs['input_ids'] == tokenizer.mask_token_id) * (inputs['labels'] == -100))].view(64, -1, 768)
marked_hidden.size()


# In[119]:


last_hidden_states = all_hidden_states[-1]


# In[157]:


_ = model.eval()
interpretable_embedding = configure_interpretable_embedding_layer(model, 'roberta.embeddings')

inputs = trainer._prepare_inputs(inputs, model)
input_ids, token_type_ids, position_ids, attention_mask, labels =     inputs['input_ids'], inputs['token_type_ids'], inputs['position_ids'], inputs['attention_mask'], inputs['labels']
input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
all_tokens = normalize_tokens(tokenizer.convert_ids_to_tokens(input_ids[0]))


# In[164]:


mask_id = 0
layer_attrs = []
for i in range(model.config.num_hidden_layers):
    lc = LayerIntegratedGradients(mlm_fwd_fn, model.roberta.encoder.layer[i])
    attributions = lc.attribute(inputs=input_embeddings, 
                                additional_forward_args=(token_type_ids, position_ids, attention_mask, labels, mask_id), 
                                n_steps=20)[0]
    layer_attrs.append(summarize_attributions(attributions).cpu().detach().tolist())


# In[165]:


fig, ax = plt.subplots(figsize=(15,5))
xticklabels=all_tokens
yticklabels=list(range(model.config.num_hidden_layers))
ax = sns.heatmap((np.array(layer_attrs) * 100).astype('int64'), xticklabels=xticklabels, yticklabels=yticklabels, annot=True, fmt='d', linewidth=0.2)
# plt.xlabel('Tokens')
# plt.ylabel('Layers')
plt.show()


# In[171]:


remove_interpretable_embedding_layer(model, interpretable_embedding)


# In[127]:


_ = model.eval()
inputs = trainer._prepare_inputs(inputs, model)
with torch.no_grad():
    loss, logits, all_hidden_states, all_attentions = model(**inputs, output_hidden_states=True, output_attentions=True)

attn_scores, attn_probs = zip(*all_attentions)
# attn_scores, attn_probs = torch.stack(attn_scores, dim=0), torch.stack(attn_probs, dim=0)
# attn_probs = attn_probs.cpu()


# In[81]:


sample_indices = [[0, 1], 
                  [2, 3]]
n_rows, n_cols = len(sample_indices), len(sample_indices[0])
fig, axs = plt.subplots(n_rows, n_cols, sharey=False, figsize=(4 * 2 * n_cols, 4 * 2 * n_rows))
sep_id = tokenizer._convert_token_to_id('Ġ,')
for row in range(n_rows):
    for col in range(n_cols):
        i, ax = sample_indices[row][col], axs[row][col]
        tokens = normalize_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
        
        pos_attn = attn_probs[:, i]
        pos_attn = pos_attn * (pos_attn > 0.3)
#         pos_attn[0] = 0  # layer 0
#         pos_attn[1, 1] = 0  # ->pos-2
        pos_attn = pos_attn.mean(dim=(0, 1))
        input_ids = inputs['input_ids'][i]
        sep_pos = (input_ids == sep_id).nonzero()[0].item()
        segment_ids = torch.zeros_like(input_ids)
        segment_ids[sep_pos + 1:] = 1
        segment_mask = segment_ids.unsqueeze(0) != segment_ids.unsqueeze(1)
        pos_attn = pos_attn * segment_mask.cpu()
        seq_len = inputs['attention_mask'][i].sum().item()
        pos_attn[:, [0, 2, seq_len - 1]] = 0
        k = torch.arange(1, pos_attn.size(0) - 1)
        pos_attn[k, k] = 0
        pos_attn[k, k - 1] = 0
        pos_attn[k, k + 1] = 0
        ax = sns.heatmap((pos_attn * 100).long(), square=True, cbar=False, annot=False, fmt='d', 
                         xticklabels=tokens, yticklabels=tokens, ax=ax)
        
#         p, h = inputs['head_mask'][i]
#         p, h = p.item(), h.item()
#         h = h - 4
#         p = p - 1
#         pos_attn = attn_probs[:, i, :, h, p]
#         pos_attn[-1, -1] = 1.
#         ax = sns.heatmap((pos_attn * 100).long(), square=True, cbar=False, annot=True, fmt='d', ax=ax)
#         ax.tick_params(top=True, labeltop=True)
#         _ = ax.set_xlabel('%s - %s' % (tokens[p], tokens[h]))


# In[78]:


i = 0
layer, head = 6, 10
seq_len = inputs['attention_mask'][i].sum().item()
attn =  attn_probs[layer, i, head, : seq_len, : seq_len]
tokens = normalize_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])[: seq_len])
size = round(attn.size(0) / 3)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(size  * 2., size), gridspec_kw={'width_ratios': [3, 1]})
_ = sns.heatmap((attn * 100).long(), square=True, cbar=True, annot=False, fmt='d', xticklabels=tokens, yticklabels=tokens, ax=ax0)
plot_head_attn(attn, tokens, ax1=ax1, marked_positions=inputs['head_mask'][i])


# In[ ]:



"""
