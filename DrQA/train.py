import re
import os
import sys
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import pandas as pd
from drqa.model import DocReaderModel
from drqa.utils import str2bool

parser = argparse.ArgumentParser(
    description='Train a Document Reader model.'
)
# system
parser.add_argument('--log_file', default='output.log',
                    help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=3,
                    help='log model loss per x updates (mini-batches).')
parser.add_argument('--data_file', default='SQuAD/data.msgpack',
                    help='path to preprocessed data file.')
parser.add_argument('--model_dir', default='models',
                    help='path to store saved models.')
parser.add_argument('--save_last_only', action='store_true',
                    help='only save the final models.')
parser.add_argument('--eval_per_epoch', type=int, default=1,
                    help='perform evaluation per x epochs.')
parser.add_argument('--seed', type=int, default=937,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
# training
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-rs', '--resume', default='',
                    help='previous model file name (in `model_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')
parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                    help='reduce initial (resumed) learning rate by this factor.')
parser.add_argument('-op', '--optimizer', default='adamax',
                    help='supported optimizer: adamax, sgd')
parser.add_argument('-gc', '--grad_clipping', type=float, default=20)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='only applied to SGD.')
parser.add_argument('-mm', '--momentum', type=float, default=0,
                    help='only applied to SGD.')
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                    help='finetune top-x embeddings.')
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
parser.add_argument('--rnn_padding', action='store_true',
                    help='perform rnn padding (much slower but more accurate).')
# model
parser.add_argument('--question_merge', default='self_attn')
parser.add_argument('--doc_layers', type=int, default=5)
parser.add_argument('--question_layers', type=int, default=5)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_features', type=int, default=4)
parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                    help='use pos tags as a feature.')
parser.add_argument('--pos_size', type=int, default=56,
                    help='how many kinds of POS tags.')
parser.add_argument('--pos_dim', type=int, default=56,
                    help='the embedding dimension for POS tags.')
parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=True,
                    help='use named entity tags as a feature.')
parser.add_argument('--ner_size', type=int, default=19,
                    help='how many kinds of named entity tags.')
parser.add_argument('--ner_dim', type=int, default=19,
                    help='the embedding dimension for named entity tags.')
parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                    const=True, default=False)
parser.add_argument('--dropout_emb', type=float, default=0.5)
parser.add_argument('--dropout_rnn', type=float, default=0.2)
parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--max_len', type=int, default=15)
parser.add_argument('--rnn_type', default='lstm',
                    help='supported types: rnn, gru, lstm')

args = parser.parse_args()

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set random seed
seed = args.seed if args.seed >= 0 else int(random.random()*1000)
print ('seed:', seed)
random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def main():
    log.info('[program starts.]')
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(model_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        for i in range(checkpoint['epoch']):
            random.shuffle(list(range(len(train))))  # synchronize random seed
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1

    if args.cuda:
        model.cuda()

    if args.resume:
        batches = BatchGen(dev, batch_size=1, evaluation=True, gpu=args.cuda)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch))
        em, f1 = score(predictions, dev_y)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        best_val_score = f1
    else:
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warn('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        # eval
        if epoch % args.eval_per_epoch == 0:
            batches = BatchGen(dev, batch_size=1, evaluation=True, gpu=args.cuda)
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch))
            em, f1 = score(predictions, dev_y)
            log.warn("dev EM: {} F1: {}".format(em, f1))
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join(model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {}]'.format(lr_decay))
    return optimizer


def load_data(opt):
    with open('SQuAD/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    if not opt['fix_embeddings']:
        embedding[1] = torch.normal(means=torch.zeros(opt['embedding_dim']), std=1.)
    with open(args.data_file, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train_orig = pd.read_csv('SQuAD/train.csv')
    dev_orig = pd.read_csv('SQuAD/dev.csv')
    train = list(zip(
        data['trn_context_ids'],
        data['trn_context_features'],
        data['trn_context_tags'],
        data['trn_context_ents'],
        data['trn_question_ids'],
        train_orig['answer_start_token'].tolist(),
        train_orig['answer_end_token'].tolist(),
        data['trn_context_text'],
        data['trn_context_spans']
    ))
    dev = list(zip(
        data['dev_context_ids'],
        data['dev_context_features'],
        data['dev_context_tags'],
        data['dev_context_ents'],
        data['dev_question_ids'],
        data['dev_context_text'],
        data['dev_context_spans']
    ))
    dev_y = dev_orig['answers'].tolist()[:len(dev)]
    dev_y = [eval(y) for y in dev_y]
    return train, dev, dev_y, embedding, opt


class BatchGen:
    def __init__(self, data, batch_size, gpu, evaluation=False):
        '''
        input:
            data - list of lists
            batch_size - int
        '''
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 7
            else:
                assert len(batch) == 9

            context_len = max(len(x) for x in batch[0])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[0]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[1][0][0])
            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                context_tag[i, :len(doc)] = torch.LongTensor(doc)

            context_ent = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[3]):
                context_ent[i, :len(doc)] = torch.LongTensor(doc)
            question_len = max(len(x) for x in batch[4])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[4]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            if not self.eval:
                y_s = torch.LongTensor(batch[5])
                y_e = torch.LongTensor(batch[6])
            text = list(batch[-2])
            span = list(batch[-1])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1

if __name__ == '__main__':
    main()
