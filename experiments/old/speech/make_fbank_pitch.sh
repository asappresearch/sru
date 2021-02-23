#!/bin/bash
# Copyright 2014  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

. ./cmd.sh
. ./path.sh
# . ./setup_*.sh ## change setup according to language

train_src=data/train_nodup
dev_src=data/eval2000

# fbank features,
train=data/train_nodup_fbank
dev=data/eval2000_fbank
fbankdir=fbank

# optional settings
stage=0

set -euxo pipefail

train_cmd=run.pl

# Make the kaldi FBANK+PITCH features,
if [ $stage -le 0 ]; then
  # Dev set
  if [ ! -e $dev ]; then
    (mkdir -p $dev; cp $dev_src/* $dev/ || true; rm -f $dev/{feats,cmvn}.scp)
    steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
       $dev $dev/log $fbankdir || exit 1;
    steps/compute_cmvn_stats.sh $dev $dev/log $fbankdir || exit 1;
  fi
  # Training set
  if [ ! -e $train ]; then
    (mkdir -p $train; cp $train_src/* $train/ || true; rm -f $train/{feats,cmvn}.scp)
    steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd -tc 10" \
       $train $train/log $fbankdir || exit 1;
    steps/compute_cmvn_stats.sh $train $train/log $fbankdir || exit 1;
  fi
  # # Split to training 90%, cv 10%
  # utils/subset_data_dir_tr_cv.sh $train ${train}_tr90 ${train}_cv10 || exit 1;
fi

echo $0 successs.
exit 0 

