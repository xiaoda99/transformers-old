from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import argparse
from tqdm import tqdm, trange
import itertools

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam

from child_generator import make_sentences
from child_frames import frames

from torch.utils.data import Dataset
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def rejoin_tokens(tokens):
    new_tokens = []
    while len(tokens) > 0:
        t = tokens.pop(0)
        if t == "[":
            t1 = tokens.pop(0)
            t2 = tokens.pop(0)
            assert t2 == "]", t2
            new_tokens.append(t + t1 + t2)
        else:
            new_tokens.append(t)
    return new_tokens


class CHILDDataset(Dataset):
    def __init__(self, tokenizer, one_sent=False, max_seq_len=None, dev_percent=-1):
        self.tokenizer = tokenizer
        self.one_sent = one_sent
        self.max_seq_len = max_seq_len

        if dev_percent == -1:
            causal_lines, turning_lines, subs_lines = [], [], []
            for frame in frames:
                causal_sent, turning_sent, subs_sent = make_sentences(**frame)
                causal_lines += causal_sent
                turning_lines += turning_sent
                subs_lines += subs_sent
            train_lines = causal_lines + turning_lines
            dev_lines = list(set(subs_lines) - set(train_lines))
            self.all_lines = train_lines + dev_lines
            self.n_dev = len(dev_lines)
        else:
            self.all_lines = list(itertools.chain.from_iterable(
                [make_sentences(**frame)[-1] for frame in frames]))
            random.shuffle(self.all_lines)
            self.n_dev = int(round(len(self.all_lines) * dev_percent))

        n_all = len(self.all_lines)
        self.n_train = n_all - self.n_dev

        if type(self.all_lines[0]) == list:
            n_substitutes = len(self.all_lines[0])
            assert all(len(substitutes) == n_substitutes for substitutes in self.all_lines)
            print('flattening all_lines: %d * %d = %d' %
                (n_all, n_substitutes, n_all * n_substitutes))
            self.all_lines = list(itertools.chain.from_iterable(self.all_lines))
            self.n_dev *= n_substitutes
            self.n_train *= n_substitutes

        self.examples = []
        cur_id = 0
        for line in self.all_lines:
            t1, t2, is_next_label = self.split_sent(line)

            tokens_a = rejoin_tokens(self.tokenizer.tokenize(t1))
            tokens_b = rejoin_tokens(self.tokenizer.tokenize(t2)) if t2 is not None else None

            example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)
            self.examples.append(example)
            cur_id += 1

        if self.max_seq_len is None:
            self.max_seq_len = max([len(example.tokens_a) + len(example.tokens_b) + 3
                if example.tokens_b is not None else len(example.tokens_a) + 2
                for example in self.examples])
            print('max_seq_len =', self.max_seq_len)

        self.features = [convert_example_to_features(example, self.max_seq_len, self.tokenizer) for example in self.examples]

    def split_sent(self, line):
        label = 0
        if "|||" in line:
            t1, t2 = [t.strip() for t in line.split("|||")]
            assert len(t1) > 0 and len(t2) > 0, "%d %d" % (len(t1), len(t2))
            if self.one_sent:
                t1 = t1 + " " + t2
                t2 = None
        else:
            assert self.one_sent
            t1, t2 = line.strip(), None
        return t1, t2, label

    def get_train_examples(self):
        return self.examples[:self.n_train]

    def get_dev_examples(self):
        return self.examples[self.n_train:]

    def get_train_features(self):
        return self.features[:self.n_train]

    def get_dev_features(self):
        return self.features[self.n_train:]

    def __len__(self):
        return len(self.all_lines)


class InputExample(object):
    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


def mask_word(tokens, tokenizer):
    output_label = []

    for i, token in enumerate(tokens):
        if token.startswith("[") and token.endswith("]"):  # masked word
            token = token[1:-1]
            tokens[i] = "[MASK]"
            output_label.append(tokenizer.vocab[token])
        else:
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b

    t1_masked, t1_label = mask_word(tokens_a, tokenizer)
    lm_label_ids = [-1] + t1_label + [-1]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b is not None and len(tokens_b) > 0:
        t2_masked, t2_label = mask_word(tokens_b, tokenizer)
        lm_label_ids += (t2_label + [-1])

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

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

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if example.guid < -5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))
        logger.info("Is next sentence label: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_percent",
                        default=-1,
                        type=float,
                        help="")
    parser.add_argument("--one_sent",
                        action='store_true',
                        help="")

    ## Required parameters
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    BERT_DIR = '/nas/pretrain-bert/pretrain-tensorflow/uncased_L-12_H-768_A-12/'
    tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_DIR, 'vocab.txt'), do_lower_case=args.do_lower_case)

    #train_examples = None
    num_train_steps = None
    if args.do_train:
        print("Generating CHILD Dataset")
        child_dataset = CHILDDataset(tokenizer, one_sent=args.one_sent, dev_percent=args.dev_percent)
        train_features = child_dataset.get_train_features()
        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForMaskedLM.from_pretrained(BERT_DIR)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)

    def validate(model, eval_dataloader):
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, lm_label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            lm_label_ids = lm_label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, lm_label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        result = {'loss': eval_loss,
                  'acc': eval_accuracy}

        # logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            # logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %.3f" % (key, result[key]), end='')


    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_lm_label_ids = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long)
        all_is_next = torch.tensor([f.is_next for f in train_features], dtype=torch.long)
        train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lm_label_ids, all_is_next)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on file.__next__
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.do_eval:
            eval_features = child_dataset.get_dev_features()
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_lm_label_ids = torch.tensor([f.lm_label_ids for f in eval_features], dtype=torch.long)
            all_is_next = torch.tensor([f.is_next for f in eval_features], dtype=torch.long)
            eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lm_label_ids, all_is_next)

            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # logger.info("Epoch 0. Evaluating on train set...")
            print("Epoch 0. Train:", end='')
            validate(model, train_dataloader)
            # logger.info("Evaluating on valid set...")
            print(" Valid:", end='')
            validate(model, eval_dataloader)
            print()

        # for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                loss = model(input_ids, segment_ids, input_mask, lm_label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if args.do_eval:
                # logger.info("Epoch %d. Evaluating on train set..." % (epoch + 1))
                print("Epoch %d. Train:" % (epoch + 1), end='')
                validate(model, train_dataloader)
                # logger.info("Evaluating on valid set...")
                print(" Valid:", end='')
                validate(model, eval_dataloader)
                print()

        # Save a trained model
        # logger.info("** ** * Saving fine - tuned model ** ** * ")
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        # if args.do_train:
        #     torch.save(model_to_save.state_dict(), output_model_file)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=-1)
    return np.all((outputs == labels) | (labels == -1), axis=-1).sum()


if __name__ == "__main__":
    main()
