
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

MAX_CNT = 10000000
rows = [None, None, None]
with torch.no_grad():
    cnt = 0
    prob = nn.Softmax()
    with open(sys.argv[2]) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            cnt += 1
            rows = rows[1:] + [row]
            if rows[0] is None or rows[0]['is_from_customer'] == 'f': continue
            if rows[1] is None or rows[1]['is_from_customer'] == 't': continue
            if '{Delayed_' in rows[0]['text']: continue
            text = dataloader.clean_str(rows[0]['text'], True)
            if len(text.split()) < 4: continue
            x = build_model_input(text, model)
            y = prob(model(x))
            score = y.squeeze()[1].item()
            if score >= 0.8:
                print("{}\t{}\t{}\t{}\t{}".format(
                    cnt-2,
                    score,
                    rows[0]['text'].strip(),
                    rows[1]['text'].strip(),
                    1
                ))
            if score >= 0.8 and rows[2]['is_from_customer'] == 'f':
                print("{}\t{}\t{}\t{}\t{}".format(
                    cnt-2,
                    score,
                    rows[0]['text'].strip(),
                    rows[2]['text'].strip(),
                    2
                ))
            if cnt % 100 == 0:
                sys.stderr.write("\r{}".format(cnt))
            if cnt == MAX_CNT: break
sys.stderr.write("\n")

