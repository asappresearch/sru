import gzip
import os
import sys
import re
import random

import numpy as np
import torch

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def read_corpus(path, clean=True, TREC=False, encoding='utf8'):
    data = []
    labels = []
    with open(path, encoding=encoding) as fin:
        for line in fin:
            label, sep, text = line.partition(' ')
            label = int(label)
            text = clean_str(text.strip()) if clean else text.strip()
            labels.append(label)
            data.append(text.split())
    return data, labels

def read_MR(path, seed=1234):
    file_path = os.path.join(path, "rt-polarity.all")
    data, labels = read_corpus(file_path, encoding='latin-1')
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_SUBJ(path, seed=1234):
    file_path = os.path.join(path, "subj.all")
    data, labels = read_corpus(file_path, encoding='latin-1')
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_CR(path, seed=1234):
    file_path = os.path.join(path, "custrev.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_MPQA(path, seed=1234):
    file_path = os.path.join(path, "mpqa.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_TREC(path, seed=1234):
    train_path = os.path.join(path, "TREC.train.all")
    test_path = os.path.join(path, "TREC.test.all")
    train_x, train_y = read_corpus(train_path, TREC=True, encoding='latin-1')
    test_x, test_y = read_corpus(test_path, TREC=True, encoding='latin-1')
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    return train_x, train_y, test_x, test_y

def read_SST(path, seed=1234):
    train_path = os.path.join(path, "stsa.binary.phrases.train")
    valid_path = os.path.join(path, "stsa.binary.dev")
    test_path = os.path.join(path, "stsa.binary.test")
    train_x, train_y = read_corpus(train_path, False)
    valid_x, valid_y = read_corpus(valid_path, False)
    test_x, test_y = read_corpus(test_path, False)
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def cv_split(data, labels, nfold, test_id):
    assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
    lst_x = [ x for i, x in enumerate(data) if i%nfold != test_id ]
    lst_y = [ y for i, y in enumerate(labels) if i%nfold != test_id ]
    test_x = [ x for i, x in enumerate(data) if i%nfold == test_id ]
    test_y = [ y for i, y in enumerate(labels) if i%nfold == test_id ]
    perm = list(range(len(lst_x)))
    random.shuffle(perm)
    M = int(len(lst_x)*0.9)
    train_x = [ lst_x[i] for i in perm[:M] ]
    train_y = [ lst_y[i] for i in perm[:M] ]
    valid_x = [ lst_x[i] for i in perm[M:] ]
    valid_y = [ lst_y[i] for i in perm[M:] ]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def cv_split2(data, labels, nfold, valid_id):
    assert (nfold > 1) and (valid_id >= 0) and (valid_id < nfold)
    train_x = [ x for i, x in enumerate(data) if i%nfold != valid_id ]
    train_y = [ y for i, y in enumerate(labels) if i%nfold != valid_id ]
    valid_x = [ x for i, x in enumerate(data) if i%nfold == valid_id ]
    valid_y = [ y for i, y in enumerate(labels) if i%nfold == valid_id ]
    return train_x, train_y, valid_x, valid_y

def pad(sequences, pad_token='<pad>', pad_left=True):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(5,max(len(seq) for seq in sequences))
    if pad_left:
        return [ [pad_token]*(max_len-len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token]*(max_len-len(seq)) for seq in sequences ]


def create_one_batch(x, y, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda()


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]

    sum_len = 0.0
    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id)
        sum_len += len(bx)
        batches_x.append(bx)
        batches_y.append(by)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]

    sys.stdout.write("{} batches, avg len: {:.1f}\n".format(
        nbatch, sum_len/nbatch
    ))

    return batches_x, batches_y


def load_embedding_npz(path):
    data = np.load(path)
    return [ w.decode('utf8') for w in data['words'] ], data['vals']

def load_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = [ ]
    vals = [ ]
    with file_open(path) as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip()
            if line:
                parts = line.split(' ')
                words.append(parts[0])
                vals += [ float(x) for x in parts[1:] ]
    return words, np.asarray(vals).reshape(len(words),-1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)
