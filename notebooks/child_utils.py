import sys
import os
from collections import defaultdict, OrderedDict, Counter
import string
from random import choice, choices, shuffle, sample, randint

from transformers import GPT2Tokenizer
cache_dir = '/nas/xd/.cache/torch/transformers/'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

cardinals = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
ordinals = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth']
digit2cardinal = OrderedDict(zip(string.digits, cardinals))
digit2ordinal = OrderedDict(zip(string.digits, ordinals))

upper_letters = [l for l in string.ascii_uppercase if len(tokenizer.tokenize('%s %s' % (l*2, l*2))) == 2]
lower_letters = [l for l in string.ascii_lowercase if len(tokenizer.tokenize('%s %s' % (l.upper()*2, l.upper()*2))) == 2]
digits = list(string.digits[1:])

def inc(token):
    assert len(token) == 1 or token in ['->'], token
    if token.isalpha(): return chr(ord(token) + 1)
    elif token.isdigit(): return str(int(token) + 1)
    else: return token

def identity(x): return x
def upper(x): return x.upper()
def lower(x): return x.lower()
def x10(x): return x + '0'
def d10(x): return x[0]
def prepend(token): return lambda x: token + x

def to_cardinal(x): return digit2cardinal[x]
def to_ordinal(x): return digit2ordinal[x]
def to_digit(word):
    for d, w in digit2cardinal.items():
        if w == word: return d
    for d, w in digit2ordinal.items():
        if w == word: return d

def double(x): return x.upper() * 2
def single(x): return x[0]

def to_rand_letter(x): return choice(upper_letters + lower_letters)
def to_rand_digit(x): return choice(digits)


def inc(token):
    assert len(token) == 1 or token in ['->'], token
    if token.isalpha(): return chr(ord(token) + 1)
    elif token.isdigit(): return str(int(token) + 1)
    else: return token

def identity(x): return x
def upper(x): return x.upper()
def lower(x): return x.lower()
def x10(x): return x + '0'
def d10(x): return x[0]
def prepend(token): return lambda x: token + x

def to_cardinal(x): return digit2cardinal[x]
def to_ordinal(x): return digit2ordinal[x]
def to_digit(word):
    for d, w in digit2cardinal.items():
        if w == word: return d
    for d, w in digit2ordinal.items():
        if w == word: return d

def double(x): return x.upper() * 2
def single(x): return x[0]

def to_rand_letter(x): return choice(upper_letters + lower_letters)
def to_rand_digit(x): return choice(digits)

'''
vocab = list(string.ascii_uppercase) #+ ['_'] * 16
# vocab = list(string.digits)[1:]
query_vocab = list(string.ascii_uppercase)
nrows, ncols = 12, 6
has_query, query_first = False, True
has_output = True
def map_fn(x): return x.lower()

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
try:
    for _ in range(nrows):
#         input_tokens = sample(vocab, ncols)
        input_tokens = [choice(vocab)] * ncols
        i = random.randint(2, len(input_tokens) - 1)
        input_tokens[i] = choice(list(set(vocab) - {input_tokens[i]})) #'*' + input_tokens[i]
    #     input_tokens[-1] = input_tokens[0]
    #     special = choice(vocab)
    #     input_tokens = [choice(vocab).lower()] * (ncols - 1) + [special]
    #     shuffle(input_tokens)
        if not query_first: print(' '.join(input_tokens), end='')
        if has_query:
    #         query_tokens = sample(input_tokens, ncols - 1)
    #         query_tokens = input_tokens.copy()
            i = random.randint(0, len(input_tokens) - 2)
            query_tokens = [input_tokens[i]]
    #         query_tokens[i] = choice(vocab)
    #         query_tokens = [t.lower() for t in query_tokens]
#             print(',', ' '.join(query_tokens), end='')
            print('After', ' '.join(query_tokens), end=', ')
        if query_first: print(' '.join(input_tokens), end='')
        print(' -> ', end='')
        if has_output:
            ans_fn = identity #upper
            output_tokens = ans_fn(input_tokens[i])
    #         output_tokens = special.lower()
    #         output_tokens = choice(input_tokens)
    #         output_tokens = list(set(input_tokens) - set(query_tokens))
    #         output_tokens = map(map_fn, input_tokens)
    #         output_tokens = reverse(input_tokens)
    #         output_tokens = query_tokens[i].lower()
            print(''.join(output_tokens), end='')
        print()
finally: sys.stdout = old_stdout
example = mystdout.getvalue()
print(example)
texts['tmp'] = '\n' + example[:-1]
if has_query: ncols += 3



def vocab_fn(mix=False): return ''.join([v[0] for v in vocabs]) if mix else choice([v[0] for v in vocabs])

def input_fn(vocab, ncols):
    if vocab is None: vocab = vocab_fn()
    return sample(vocab, ncols)
#     return [choice(string.ascii_uppercase)] * ncols
def i_fn(ncols, i_range=[0, 0]):
    return choice(range(0 + i_range[0], ncols + i_range[1])) if type(i_range) in [list] else i_range

def get_tgt(token, tgt_vocab, tgt_fn=identity):
    if tgt_vocab is None: return tgt_fn(token)
    return choice(tgt_vocab)
#     return choice(list(set(tgt_vocab) - {token}))

def update_input(input_tokens, i, tgt, updates=[]):
    for offset, val in updates: input_tokens[i + offset] = val

def gen_query(input_tokens, i):
    return None, None
    return [token for j, token in enumerate(input_tokens) if j != i], None
    return [input_tokens[i - 1]], None

def get_ans_fn(vocab):
    return identity
    fns = []
    vocab = set(vocab)
    if vocab.issubset(set(string.ascii_uppercase)): fns += [double]#, lower]
    if vocab.issubset(set(string.ascii_lowercase)): fns += [double]#, upper]
    if vocab.issubset(set(string.digits)):
#         fns += [prepend('0'), prepend('1'), double, x10]
        fns += [to_word, to_ordinal]
    if len(fns) == 0: fns =[identity]
    return choice(fns)

task_configs = {
    'ith': {'i_range': 1},
    'after bracket': {'i_range': [1, 0], 'updates': [(-1, '(')]},
    'in brackets': {'i_range': [1, -1], 'updates': [(-1, '('), (1, ')')]},
    'special type': {'vocab': vocabs[0][0], 'tgt_vocab': vocabs[1][0], 'ans_fn': upper},
    'special': {'i_range': [2, 0], 'use_extended_vocab': True, 'vocab_per_row': True}
}

def gen_example(nrows=16, ncols=3, use_extended_vocab=False, vocab_per_row=False,
                vocab=None, tgt_vocab=None, tgt_fn=identity, ans_fn=identity, 
                i_range=[0, 0], updates=[]):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
#         vocab = vocab_fn()
        ans_fn = ans_fn or get_ans_fn(tgt_vocab)
        if use_extended_vocab and not vocab_per_row:
            vocab, tgt_fn, ans_fn = get_extended_vocab_and_fns(base_vocab=None, fn=None, tgt_fn1=None)#, ans_fn=None)
        for _ in range(nrows):
            if use_extended_vocab and vocab_per_row:
                vocab, tgt_fn, ans_fn = get_extended_vocab_and_fns(base_vocab=None, fn=None, tgt_fn1=None)#, ans_fn=None)
#             random.seed(datetime.now())
            input_tokens = input_fn(vocab, ncols)
            i = i_fn(ncols, i_range=i_range)
            tgt = get_tgt(input_tokens[i], tgt_vocab, tgt_fn=tgt_fn)
            input_tokens[i] = tgt
            update_input(input_tokens, i, tgt, updates=updates)
            ans = ans_fn(tgt)
            query = gen_query(input_tokens, i)
            if query[0]: print(' '.join(query[0]), end=', ')
    #         print()
            print(' '.join(input_tokens), end='')
            if query[1]: print(',\n' + ' '.join(query[1]), end='')
            print(' ->', end=' ')
            print(ans)
    finally: sys.stdout = old_stdout
    return mystdout.getvalue(), ans_fn

task_name = 'after bracket'
# task_name = 'in brackets'
task_name = 'ith'
# task_name = 'special type'
configs = task_configs[task_name]
# configs = list(task_configs.items())[-1][1]
nrows, ncols = 12, 4
vocab = None or upper_letters
tgt_fn, ans_fn = identity, identity
# tgt_fn, ans_fn = lower, upper
example, ans_fn = gen_example(nrows=nrows, ncols=ncols, 
                              vocab=upper_letters, 
                              tgt_fn=tgt_fn, ans_fn=ans_fn,
                              **configs)
print(example)
texts['tmp'] = '\n' + example[:-1]
'''
