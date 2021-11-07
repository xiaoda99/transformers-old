import os
import json
import itertools
from itertools import product, chain

from pytorch_pretrained_bert import tokenization, BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining, BertConfig
from child_frames import frames


CONFIG_NAME = 'bert_config.json'
BERT_DIR = '/nas/pretrain-bert/pretrain-tensorflow/uncased_L-12_H-768_A-12/'
tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_DIR, 'vocab.txt'))


A_template = "{rel_prefix} {dt} {ent0} {rel} {dt} {ent1} {rel_suffix}"
B_templates = ["{pred_prefix} {dt} {ent} {pred}", "{pred_prefix} {pred} {dt} {ent}"]

# causal_templates = [["{A} because {B}."],# "{B} so {A}."],
#                     ["{A} so {B}."],# "{B} because {A}."]
#                    ]
# turning_templates = [["{A} although {B}."],# "{B} but {A}."],
#                      ["{A} but {B}."],# "{B} although {A}."]
#                     ]

causal_templates = [["{A} ||| {conj} {B}."],# "{B} so {A}."],
                    ["{A} ||| {conj} {B}."],# "{B} because {A}."]
                   ]
turning_templates = [["{A} ||| {conj} {B}."],# "{B} but {A}."],
                     ["{A} ||| {conj} {B}."],# "{B} although {A}."]
                    ]


def reverse(l):
    return list(reversed(l))


def mask(ent_str):
    tokens = ent_str.strip().split()
    if len(tokens) == 1:
        return '[%s]' % tokens[0]
    elif len(tokens) == 2:
        assert tokens[0] == 'the', ent_str
        return '%s [%s]' % (tokens[0], tokens[1])
    else:
        assert False, ent_str


def make_sentences(index=-1, orig_sentence='', entities=["John", "Mary"], entity_substitutes=None, determiner="",
                   packed_relations=["rel/~rel", "rev_rel/~rev_rel"], packed_relation_substitutes=None,
                   relation_prefix="", relation_suffix="",
                   packed_predicates=["pred0/~pred0", "pred1/~pred1"], predicate_prefix="", prepositive_pred=False,
                   predicate_dichotomy=True, reverse_causal=False, conjunctions=[["because", "so"], ["although", "but"]]):
    assert entities[0].lower() in tokenizer.vocab , entities[0]
    assert entities[1].lower() in tokenizer.vocab , entities[1]

    def form_As(packed_rels):
        relations, neg_relations = zip(*[rel.split("/") for rel in packed_rels])
        relations, neg_relations = list(relations), list(neg_relations)

        As = [A_template.format(dt=determiner, ent0=ent0, ent1=ent1, rel=rel, rel_prefix=relation_prefix, rel_suffix=relation_suffix)
              for ent0, ent1, rel in [entities + relations[:1], reverse(entities) + reverse(relations)[:1]]]
        negAs = [A_template.format(dt=determiner, ent0=ent0, ent1=ent1, rel=rel, rel_prefix=relation_prefix, rel_suffix=relation_suffix)
                 for ent0, ent1, rel in [entities + neg_relations[:1], reverse(entities) + reverse(neg_relations)[:1]]]
        return As, negAs

    As, negAs = form_As(packed_relations)

    substituted_As, substituted_negAs = [], []
    for packed_rel_subs in zip(*packed_relation_substitutes):
        subs_As, subs_negAs = form_As(packed_rel_subs)
        substituted_As += subs_As
        substituted_negAs += subs_negAs

    if "/" in packed_predicates[0]:
        predicates, neg_predicates = zip(*[pred.split("/") for pred in packed_predicates])
        predicates, neg_predicates = list(predicates), list(neg_predicates)
    else:
        predicates, neg_predicates = packed_predicates, []

    B_template = B_templates[int(prepositive_pred)]
    Bs = [B_template.format(dt=determiner, ent=mask(ent), pred=pred, pred_prefix=predicate_prefix)
          for ent, pred in zip(entities, predicates)]
    negBs = [B_template.format(dt=determiner, ent=mask(ent), pred=pred, pred_prefix=predicate_prefix)
             for ent, pred in zip(entities, neg_predicates)]
    if predicate_dichotomy:
        Bs += [B_template.format(dt=determiner, ent=mask(ent), pred=pred, pred_prefix=predicate_prefix)
               for ent, pred in zip(entities, reversed(neg_predicates))]
        negBs += [B_template.format(dt=determiner, ent=mask(ent), pred=pred, pred_prefix=predicate_prefix)
                  for ent, pred in zip(entities, reversed(predicates))]

    def form_sentences(sentence_template, As, Bs, conj):
        return [" ".join(sentence_template.format(A=A, B=B, conj=conj).split()) for A, B in product(As, Bs)]

    def form_all_sentences(As, negAs, Bs, negBs):
        causal_sentences = []
        causal_conj = conjunctions[0][int(reverse_causal)]
        for causal_template in causal_templates[int(reverse_causal)]:
            for A, B in [(As, Bs), (negAs, negBs)]:
                causal_sentences += form_sentences(causal_template, A, B, causal_conj)

        turning_sentences = []
        turning_conj = conjunctions[1][int(reverse_causal)]
        for turning_template in turning_templates[int(reverse_causal)]:
            for A, B in [(As, negBs), (negAs, Bs)]:
                turning_sentences += form_sentences(turning_template, A, B, turning_conj)

        sentences = causal_sentences + turning_sentences
        return sentences, causal_sentences, turning_sentences

    sentences, causal_sentences, turning_sentences = form_all_sentences(As, negAs, Bs, negBs)
    # substituted_sentences = sentences

    if packed_relation_substitutes is not None:
        substituted_sentences = form_all_sentences(substituted_As, substituted_negAs, Bs, negBs)[0]

    substituted_sent_groups = list(zip(sentences, substituted_sentences))

    if entity_substitutes is not None:
        for sub in entity_substitutes:
            for ent in sub:
                assert ent.lower() in tokenizer.vocab , ent + " not in BERT vocab"
        assert len(set(chain.from_iterable(entity_substitutes))) == 4, entity_substitutes
        assert len(set(chain.from_iterable(entity_substitutes)).union(set(entities))) == 6

        entity_substitutes = list(itertools.product(entities[:1] + entity_substitutes[0], entities[1:] + entity_substitutes[1]))
        substituted_sent_groups = [[sent.replace(entities[0], sub[0]).replace(entities[1], sub[1]) 
            for sent in sent_group for sub in entity_substitutes] for sent_group in substituted_sent_groups]
    return causal_sentences, turning_sentences, substituted_sent_groups