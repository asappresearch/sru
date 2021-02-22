
import sys
import csv
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sru import *
import dataloader
import modules
from train_classifier import Model

def build_model(args, emb_layer):
#    train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path)
#    data = train_x + valid_x + test_x
#    emb_layer = modules.EmbeddingLayer(
#        args.d, data,
#        embs = dataloader.load_embedding(args.embedding)
#    )
    model = Model(args, emb_layer, 2)
    return model

def build_model_input(text, model):
    x, y = dataloader.create_one_batch(
            [text.split()],
            [0],
            model.emb_layer.word2id
    )
    return x

args, model_states, emb_layer = torch.load(sys.argv[1])
model = build_model(args, emb_layer)
model.load_state_dict(model_states)
model.eval()

MAX_CNT = 2000000
freq = Counter()
score = Counter()
with torch.no_grad():
    cnt = 0
    prob = nn.Softmax()
    with open(sys.argv[2]) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if row['is_from_customer'] == 'f': continue
            if '{Delayed_' in row['text']: continue
            raw_text = row['text'].strip()
            text = dataloader.clean_str(row['text'], True)
            if len(text.split()) < 4: continue
            x = build_model_input(text, model)
            y = prob(model(x))
            if raw_text in freq:
                freq[raw_text] += 1
            else:
                freq[raw_text] = 1
                score[raw_text] = y.squeeze()[1].item()
            cnt += 1
            if cnt % 100 == 0:
                sys.stderr.write("\r{},{}".format(cnt, len(freq)))
            if cnt == MAX_CNT: break
sys.stderr.write("\n")

combined = set([text for text, val in freq.most_common(5000) + score.most_common(5000)])
for text in combined:
    if score[text] < 0.7: continue
    print ("{}\t{}\t{}".format(
        text,
        freq[text],
        score[text]
    ))
