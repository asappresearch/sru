
import os
import sys
import argparse
import time
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import cuda_functional as MF
import dataloader
import modules

class Model(nn.Module):
    def __init__(self, args, emb_layer, nclasses=2):
        super(Model, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.emb_layer = emb_layer
        if args.cnn:
            self.encoder = modules.CNN_Text(
                emb_layer.n_d,
                widths = [3,4,5]
            )
            d_out = 300
        elif args.lstm:
            self.encoder = nn.LSTM(
                emb_layer.n_d,
                args.d,
                args.depth,
                dropout = args.dropout,
            )
            d_out = args.d
        else:
            self.encoder = MF.SRU(
                emb_layer.n_d,
                args.d,
                args.depth,
                dropout = args.dropout,
                use_tanh = 1,
            )
            d_out = args.d
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        if self.args.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        emb = self.drop(emb)

        if self.args.cnn:
            output = self.encoder(emb)
        else:
            output, hidden = self.encoder(emb)
            output = output[-1]

        output = self.drop(output)
        return self.out(output)

def eval_model(niter, model, valid_x, valid_y):
    model.eval()
    N = len(valid_x)
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0
    for x, y in zip(valid_x, valid_y):
        x, y = Variable(x, volatile=True), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.data[0]*x.size(1)
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data).cpu().sum()
        cnt += y.numel()
    model.train()
    return 1.0-correct/cnt

def train_model(epoch, model, optimizer, train_x, train_y, valid_x, valid_y, test_x, test_y, best_valid, test_err):
    start = time.time()
    model.train()
    args = model.args
    N = len(train_x)
    niter = epoch*len(train_x)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for x, y in zip(train_x, train_y):
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    valid_err = eval_model(niter, model, valid_x, valid_y)
    test_err = eval_model(niter, model, test_x, test_y)
    stop = time.time()
    if (((epoch % 10) == 0) or (epoch == args.max_epoch)):
        sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_err={:.6f} time={:.6f}\n".format(
            epoch, niter,
            optimizer.param_groups[0]['lr'],
            loss.data[0],
            valid_err,
            (stop-start)
        ))

    return epoch, loss.data[0],valid_err, test_err, (stop-start)

def main(args):
    datasetList = ['mr', 'subj', 'cr', 'mpqa', 'trec', 'sst']
    numberOfTest = 5
    args.max_epoch = 100
    for dset in datasetList:
        if dset == 'mr':
            data, label = dataloader.read_MR(args.path)
        elif dset == 'subj':
            data, label = dataloader.read_SUBJ(args.path)
        elif dset == 'cr':
            data, label = dataloader.read_CR(args.path)
        elif dset == 'mpqa':
            data, label = dataloader.read_MPQA(args.path)
        elif dset == 'trec':
            train_x, train_y, test_x, test_y = dataloader.read_TREC(args.path)
            data = train_x + test_x
            label = None
        elif dset == 'sst':
            train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path)
            data = train_x + valid_x + test_x
            label = None
        else:
            raise Exception("unknown dataset: {}".format(dset))

        emb_layer = modules.EmbeddingLayer(
            args.d, data,
            embs = dataloader.load_embedding(args.embedding)
        )

        if dset == 'trec':
            train_x, train_y, valid_x, valid_y = dataloader.cv_split2(
                train_x, train_y,
                nfold = 10,
                valid_id = args.cv
            )
        elif dset != 'sst':
            train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.cv_split(
                data, label,
                nfold = 10,
                test_id = args.cv
            )
        nclasses = max(train_y)+1

        train_x, train_y = dataloader.create_batches(train_x, train_y, args.batch_size, emb_layer.word2id, sort = dset == 'sst')
        valid_x, valid_y = dataloader.create_batches(valid_x, valid_y, args.batch_size, emb_layer.word2id, sort = dset == 'sst')
        test_x, test_y = dataloader.create_batches(test_x, test_y, args.batch_size, emb_layer.word2id, sort = dset == 'sst')

        for models in range(3):
            if models == 1:
                args.cnn = True
                modelName = 'CNN'
            elif models == 2:
                args.cnn = False
                args.lstm = True
                modelName = 'LSTM'
            else:
                args.lstm = False
                modelName = 'SRU'

            sys.stdout.write("Training {} with {} architecture: \n".format(dset,modelName))
            args.dropout = 0.5


            for testNo in range(numberOfTest):
                model = Model(args, emb_layer, nclasses).cuda()
                need_grad = lambda x: x.requires_grad
                optimizer = optim.Adam(filter(need_grad, model.parameters()), lr = args.lr)

                best_valid = 1e+8
                test_err = 1e+8
                results = []
                for epoch in range(args.max_epoch):
                    results.append(train_model(epoch, model, optimizer, train_x, train_y, valid_x, valid_y, test_x, test_y, best_valid, test_err))
                
                with open('results_{d}_{m}_{i}.csv'.format(d=dset, m=modelName, i=(testNo+1)), 'wb') as dump:
                    wr = csv.writer(dump, delimiter=',')
                    wr.writerow(['Epoch','Training Loss', 'Validation Error', 'Test Error', 'Duration'])
                    for idx, value in enumerate(results):
                        wr.writerow(value)

            # dropOuts = [0.1, 0.3, 0.5, 0.7]
            # bestDropOut = 0.1
            # bestDropOutError = 1e+8

            # sys.stdout.write("Checking for the best Dropout! \n")
            # for checkingDropouts in dropOuts:
            #     args.dropout = checkingDropouts
            #     model = Model(args, emb_layer, nclasses).cuda()
            #     need_grad = lambda x: x.requires_grad
            #     optimizer = optim.Adam(filter(need_grad, model.parameters()), lr = args.lr)

            #     best_valid = 1e+8
            #     test_err = 1e+8
            #     results = []
            #     for epoch in range(args.max_epoch):
            #         train_loss, valid_err, test_err = train_model(epoch, model, optimizer,
            #             train_x, train_y,
            #             valid_x, valid_y,
            #             test_x, test_y,
            #             best_valid, test_err
            #         )
            #         if args.lr_decay>0:
            #             optimizer.param_groups[0]['lr'] *= args.lr_decay
            #     if test_err < bestDropOutError:
            #         bestDropOutError = test_err
            #         bestDropOut = checkingDropouts
            
            # sys.stdout.write("Best dropout for {} using {}: {:.2f}\n".format(dset, modelName, bestDropOut))





if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--dataset", type=str, default="mr", help="which dataset")
    argparser.add_argument("--path", type=str, required=True, help="path to corpus directory")
    argparser.add_argument("--embedding", type=str, required=True, help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--d", type=int, default=128)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--cv", type=int, default=0)

    args = argparser.parse_args()
    print (args)
    main(args)

