#!/usr/bin/env python
# coding: utf-8

# In[92]:


from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import random
import string
from collections import defaultdict
from itertools import product, chain
import numpy as np
from pattern.en import comparative


# In[34]:


import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from enum import Enum
from typing import List, Optional, Union

from child_frames import frames
from utils import *

import sys
sys.path.insert(0, '/nas/xd/projects/transformers/src/transformers')

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


# In[88]:


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
        
tag2id = {'Ġsame': 0, 'Ġopposite': 1, 'Ġunrelated': 2, 'Ġformer': 3, 'Ġlatter': 4, 'Ġanother': 5}
id2tag = {v: k for k, v in tag2id.items()}

def make_sentences(index=-1, entities=["_X", "_Z"], entity_set=string.ascii_uppercase, determiner="",
                   relation_group=[["big",], ["small"]], rand_relation_group=[["short"], ["tall", "high"]],
                   relation_prefix="", relation_suffix="", predicate_prefix="",
                   n_entity_trials=3, has_negA=True, has_negB=True, has_neutral=False, mask_types={'sent_rel'}, 
                   lexical_relations=['same', 'opposite', 'unrelated'], tag_lexical_rel=False, tag_entity_rel=False):
    if tag_lexical_rel: mask_types.add('lexical_rel')
    if tag_entity_rel: mask_types.add('entity_rel')
        
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
        def get_lexical_rel(rel_id_A, rel_id_B): return 'same' if rel_id_A == rel_id_B else 'opposite'
        def compare_and_tag_entity(B, A):
            entity = [e for e in entities if e in B][0]
            entity_rel = 'former' if A.strip().startswith(entity) else 'latter'
            if 'entity_rel' in mask_types: entity_rel = mask(entity_rel)
            return B.replace(entity, entity + ' ( ' + entity_rel + ' )')                    
        
        if 'sent_rel' in mask_types: conj = mask(conj)
        As_with_rel_ids = [(A, extract_rel_id(A)) for A in As]
        Bs_with_rel_ids = [(B, extract_rel_id(B)) for B in Bs]
            
        sentences = []
        for (A, rel_id_A), (B, rel_id_B) in product(As_with_rel_ids, Bs_with_rel_ids):
            lexical_rel = 'unrelated' if 'Maybe' in conj else get_lexical_rel(rel_id_A, rel_id_B)
            if lexical_rel in lexical_relations:
                if tag_entity_rel: B = compare_and_tag_entity(B, A)
                if not tag_lexical_rel: lexical_rel = ''
                elif 'lexical_rel' in mask_types: lexical_rel = mask(lexical_rel)
                sent = sentence_template.format(A=strip_rel_id(A), B=strip_rel_id(B, lexical_rel), conj=conj)
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
    assert len(sentences['Right']) == len(sentences['Wrong']), '%d %d' % (len(sentences['Right']), len(sentences['Wrong']))
    if has_neutral: sentences['Maybe'] = random.sample(sentences['Maybe'], len(sentences['Right']))
    sentences = join_lists(sentences[k] for k in (sentences.keys() if has_neutral else ['Right', 'Wrong']))
    
    substituted_sent_groups = []
    for sent in sentences:
        sent_group = []
        for _ in range(n_entity_trials):
            e0, e1 = random.sample(entity_set, 2)
            sent_group.append(sent.replace(entities[0], e0).replace(entities[1], e1))
        substituted_sent_groups.append(sent_group)
    return sentences, substituted_sent_groups

# make_sentences(has_negA=True, has_negB=True, has_neutral=False, tag_lexical_rel=True, tag_entity_rel=True, 
#                mask_types={'sent_rel'})[0]


# In[18]:


P_template = '{ent0} {rel} {ent1}'
transitive_template = '{p0} and {p1} , so {Q} ? {conj} .'
transitive_wh_QA_template = '{which} is {pred} ? {ent} .'
    
def make_transitive(entities=["_X", "_Y", "_Z"], entity_set=string.ascii_uppercase, relation_group=[["big", ], ["small", ]], 
                    n_entity_trials=3, has_negP=True, has_negQ=True, has_neutral=False, mask_types=['sent_rel']):
    def form_atoms(relations, entities, has_neg=True):
        atoms = [P_template.format(ent0=ent0, ent1=ent1, rel=rel) 
                 for ent0, ent1, rel in [entities + relations[:1], reverse(entities) + reverse(relations)[:1]]]
        if has_neg:
            neg_relations = [r.replace('is ', 'is not ') for r in relations]
            atoms += [P_template.format(ent0=ent0, ent1=ent1, rel=rel) 
                      for ent0, ent1, rel in [entities + reverse(neg_relations)[:1], reverse(entities) + neg_relations[:1]]]
        return atoms
 
    def form_sentences(transitive_template, Ps, Qs, conj):
        sentences = []
        if 'sent_rel' in mask_types: conj = mask(conj)
        for (p0, p1), Q in product(Ps, Qs):
            sent = transitive_template.format(p0=strip_rel_id(p0), p1=strip_rel_id(p1), Q=strip_rel_id(Q), conj=conj)
            sent = " " + " ".join(sent.split())
            sentences.append(sent)
        return sentences
    
    def form_all(P0_entities, P1_entities, Q_entities, neutral=False):
        P0, P1 = [], []
        for rel0 in relation_group[0]:
            for rel1 in relation_group[1]:
                relations = ["is %s:%d than" % (get_comparative(rel), i) for i, rel in enumerate([rel0, rel1])]
                P0 += form_atoms(relations, P0_entities, has_neg=has_negP)
                P1 += form_atoms(relations, P1_entities, has_neg=has_negP)
        Ps = [(p0, p1) for p0, p1 in list(product(P0, P1)) + list(product(P1, P0))]

        Qs = form_atoms(relations, Q_entities, has_neg=has_negQ)
        negQs = [swap_entities(Q, *Q_entities) for Q in Qs]
        
        for P, Q, conj in [(Ps, Qs, 'Right'), (Ps, negQs, 'Wrong')]:
            if neutral: conj = 'Maybe'
            sentences[conj] += form_sentences(transitive_template, P, Q, conj)
        return sentences
    
    e0, e1, e2 = entities
    sentences = defaultdict(list)
    form_all(P0_entities=[e0, e1], P1_entities=[e1, e2], Q_entities=[e0, e2])
    assert len(sentences['Right']) == len(sentences['Wrong']), '%d %d' % (len(sentences['Right']), len(sentences['Wrong']))
    sample_ratio = len(relation_group[0]) * len(relation_group[1])
    if sample_ratio > 1:
        for key in sentences: sentences[key] = random.sample(sentences[key], len(sentences[key]) // sample_ratio)
#     print('nRight =', len(sentences['Right']))
    if has_neutral:
        form_all(P0_entities=[e0, e1], P1_entities=[e0, e2], Q_entities=[e1, e2], neutral=True)
        sentences['Maybe'] = random.sample(sentences['Maybe'], len(sentences['Right']))
    sentences = join_lists(sentences[k] for k in (sentences.keys() if has_neutral else ['Right', 'Wrong']))
    
    substituted_sent_groups = []
    for sent in sentences:
        sent_group = []
        for _ in range(n_entity_trials):
            e0, e1, e2 = random.sample(entity_set, 3)
            sent_group.append(sent.replace(entities[0], e0).replace(entities[1], e1).replace(entities[2], e2))
        substituted_sent_groups.append(sent_group)
    return sentences, substituted_sent_groups

# make_transitive(has_negP=False, has_negQ=False, has_neutral=False)


# In[10]:


model_class, tokenizer_class, shortcut = RobertaForMaskedLM, RobertaTokenizer, 'roberta-large'
# model_class = RobertaDoubleHeadsModel
model, tokenizer = None, tokenizer_class.from_pretrained(shortcut)


# In[90]:


random.shuffle(frames)
n_entity_trials = 3
all_lines = [make_sentences(relation_group=rg, rand_relation_group=frames[(i + 1) % len(frames)], n_entity_trials=n_entity_trials, 
                            has_negA=True, has_negB=True, tag_lexical_rel=True, tag_entity_rel=True,
                            has_neutral=False, mask_types={'sent_rel'})[1] 
             for i, rg in enumerate(frames)]
# all_lines = [make_transitive(relation_group=rg, n_entity_trials=10, 
#                              has_negP=False, has_negQ=False, has_neutral=False, mask_types=['sent_rel'])[1] 
#              for i, rg in enumerate(frames)]
# all_lines = join_lists(all_lines)
# all_lines = join_lists(all_lines)
tokenizer.tag2id, tokenizer.id2tag = tag2id, id2tag
for k in CHILDDataset.all_lines: CHILDDataset.all_lines[k] = None
train_dataset = CHILDDataset(all_lines, tokenizer, has_markers=True, max_noise_len=0, split_pct=[0.7, 0.3, 0.0], mode='train')
eval_dataset = CHILDDataset(all_lines, tokenizer, has_markers=True, max_noise_len=0, split_pct=[0.7, 0.3, 0.0], mode='dev')
print('nTrain = %d, nValid = %d' % (len(train_dataset), len(eval_dataset)))


# In[91]:


model = model_class.from_pretrained('roberta-base', model=model)
steps = 100 * n_entity_trials // 3
training_args = TrainingArguments(output_dir="./models/model_name", 
    overwrite_output_dir=True, do_train=True, do_eval=True,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    learning_rate=2e-5, num_train_epochs=3,
    logging_steps=steps, eval_steps=steps, save_steps=0,
    no_cuda=False, evaluate_during_training=True,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.tokenizer = tokenizer
trainer.train()


# In[75]:


dataloader = DataLoader(
    eval_dataset,
    sampler=RandomSampler(eval_dataset),
    batch_size=trainer.args.eval_batch_size,
    collate_fn=trainer.data_collator,
    drop_last=trainer.args.dataloader_drop_last,
)


# In[76]:


for inputs in dataloader: break


# In[77]:


inputs['marked_positions'].size()


# In[78]:


i = 0
tokenizer.decode(inputs['input_ids'][i])
[tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])[j] for j in inputs['marked_positions'][i]]


# In[133]:


inputs = trainer._prepare_inputs(inputs, model)
loss, logits = model(**inputs)


# In[ ]:


model.roberta.encoder.layer[0].attention.self

