#!/bin/bash

# This script trains and evaluate LSTM models. There is no
# discriminative training yet.
# In this recipe, CNTK directly read Kaldi features and labels,
# which makes the whole pipline much simpler.

. ./cmd.sh
. ./path.sh

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
set -u           #Fail on an undefined variable

#############################################################
# Generate data and alignment from Kaldi
#############################################################


# label,
# source data,
ali_src=exp/tri4_ali_nodup
graph_src=exp/tri4/graph_sw1_tg

# features
train_src=data/train_nodup_fbank
dev_src=data/eval2000_fbank

#############################################################
# CNTK config
#############################################################

cntk_config=CNTK_SRU.config
ndl=SRU_12L.ndl
smbr_config=CNTK_SRU_smbr.config

# path to cntk binary
cn_gpu=cntk

# The device number to run the training
# change to AUTO to select the card automatically
DeviceNumber=0

# model will gets dumped here
trainlog=train_cntk

num_utts_per_iter=80
smooth_factor=0.1
use_one_sil=true

njdec=80

scoring="--min-lmwt 8 --max-lmwt 19"
stage=0

expdir=exp_cntk/SRU_12L

acwt=0.0833

. utils/parse_options.sh || exit 1;


alidir=${expdir}_ali
denlatdir=${expdir}_denlats
smbrdir=${expdir}_smbr_onesil


labelDim=$(($(cat ${ali_src}/final.occs | wc -w)-2))
baseFeatDim=$(feat-to-dim scp:${train_src}/feats.scp - | cat -)
# This effectively delays the output label by 5 frames, so that the LSTM sees 5 future frames.
featDim=$((baseFeatDim*21))
rowSliceStart=$((baseFeatDim*20))
mkdir -p $expdir

if [ ! -d ${train_src}_tr90 ] ; then
  echo "Training and validation sets not found. Generating..."
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 ${train_src} ${train_src}_tr90 ${train_src}_cv10
  echo "done."
fi

feats_tr="scp:${train_src}_tr90/feats.scp"
feats_cv="scp:${train_src}_cv10/feats.scp"
labels_tr="ark:ali-to-pdf $ali_src/final.mdl \"ark:gunzip -c $ali_src/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"

if [ $stage -le 0 ] ; then
  (feat-to-len "$feats_tr" ark,t:- > $expdir/cntk_train.counts) || exit 1;
  echo "$feats_tr" > $expdir/cntk_train.feats
  echo "$labels_tr" > $expdir/cntk_train.labels

  (feat-to-len "$feats_cv" ark,t:- > $expdir/cntk_valid.counts) || exit 1;
  echo "$feats_cv" > $expdir/cntk_valid.feats
  echo "$labels_tr" > $expdir/cntk_valid.labels

  for (( c=0; c<labelDim; c++)) ; do
    echo $c
  done >$expdir/cntk_label.mapping
fi

if [ $stage -le 1 ] ; then
  ### setup the configuration files for training CNTK models ###
  cp conf/cntk/default_macros.ndl $expdir/
  cp $cntk_config $expdir/CNTK2.config
  cp $ndl $expdir/nn.ndl
  ndlfile=$expdir/nn.ndl

  tee $expdir/Base.config <<EOF
ExpDir=$expdir
logFile=${trainlog}
modelName=cntk.nn

verbosity=0
labelDim=${labelDim}
featDim=${baseFeatDim}
labelMapping=${expdir}/cntk_label.mapping
featureTransform=NO_FEATURE_TRANSFORM

inputCounts=${expdir}/cntk_train.counts
inputFeats=${expdir}/cntk_train.feats
inputLabels=${expdir}/cntk_train.labels

cvInputCounts=${expdir}/cntk_valid.counts
cvInputFeats=${expdir}/cntk_valid.feats
cvInputLabels=${expdir}/cntk_valid.labels
EOF

  ## training command ##
  $cuda_cmd $expdir/log/cmdtrain.log \
  $cn_gpu configFile=${expdir}/Base.config configFile=${expdir}/CNTK2.config \
    DeviceNumber=$DeviceNumber action=TrainLSTM ndlfile=$ndlfile FeatDim=$featDim baseFeatDim=$baseFeatDim RowSliceStart=$rowSliceStart maxEpochs=25 uttNum=80

echo "$0 successfuly finished.. $expdir"

fi

if [ $stage -le 2 ] ; then

  config_write=conf/cntk/CNTK2_write.config
  cnmodel=$expdir/cntk.nn
  action=write
  cp $ali_src/final.mdl $expdir
  cntk_string="$cn_gpu configFile=$config_write verbosity=0 DeviceNumber=-1 modelName=$cnmodel labelDim=$labelDim featDim=$featDim action=$action ExpDir=$expdir"
  local/decode_cntk2.sh --nj 80 --cmd $decode_cmd --acwt $acwt --scoring-opts "$scoring" \
    $graph_src $dev_src $expdir/decode_$(basename $dev_src) "$cntk_string" || exit 1;
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} $dev_src \
        $expdir/decode_$(basename $dev_src){,fsh_fg}
fi

lang=data/lang
###################################
# Start discriminative training
###################################

# Alignment.
if [ $stage -le 3 ]; then
  mkdir -p $alidir
  cp -f conf/cntk/Align.config $alidir/Align.config
  cp $expdir/cntk.nn $expdir/cntk.mdl
  cp ${ali_src}/tree $expdir/
  local/align_cntk.sh --num-threads 1 --nj 60 --cmd "$train_cmd" \
    --feat-dim $featDim --device -1 \
    --cntk-config $alidir/Align.config \
    $train_src $lang $expdir $alidir || exit 1;
fi

# Denominator lattices.
if [ $stage -le 4 ]; then
  mkdir -p $denlatdir
  cp -f conf/cntk/Align.config $denlatdir/Decode.config
  local/make_denlats_cntk.sh --num-threads 1 --nj 60 \
    --feat-dim $featDim --cmd "$train_cmd" --acwt $acwt \
    --device -1 --cntk-config $denlatdir/Decode.config \
    --ngram-order 2 \
    $train_src $lang $expdir $denlatdir || exit 1;
fi

# Sequence training.
if [ $stage -le 5 ]; then
  mkdir -p $smbrdir/configs
  cp -f ${smbr_config} $smbrdir/configs/Train.config
  cp -f conf/cntk/CNTK2_smbr.mel $smbrdir/configs/edit.mel
  cp -f $ndl $smbrdir/configs/model.ndl
  cp -f conf/cntk/default_macros.ndl $smbrdir/configs/default_macros.ndl
  cntk_train_opts=""
  cntk_train_opts="$cntk_train_opts baseFeatDim=$baseFeatDim RowSliceStart=$rowSliceStart"
  cntk_train_opts="$cntk_train_opts numUttsPerMinibatch=$num_utts_per_iter "
  local/train_sequence.sh --num-threads 1 --cmd "$cuda_cmd" --momentum 0.9 \
    --learning-rate "0.000002*20" --num-iters 20 --feat-dim $featDim \
    --acwt $acwt --evaluate-period 100 --truncated true \
    --device $DeviceNumber --cntk-config $smbrdir/configs/Train.config \
    --minibatch-size 20 --cntk-train-opts "$cntk_train_opts" \
    --clipping-per-sample 0.05 --smooth-factor $smooth_factor \
    --one-silence-class ${use_one_sil} \
    $train_src $lang $expdir $alidir $denlatdir $smbrdir || exit 1;
  cd $smbrdir/cntk_model
  ln -s cntk.sequence cntk.sequence.4
  cd -
fi

# decoding
iters=( "1" "2" "3" "4" )
if [ $stage -le 6 ] ; then
  for iter in "${iters[@]}"
  do
    config_write=conf/cntk/CNTK2_write.config
    cnmodel=$smbrdir/cntk_model/cntk.sequence.$iter
    action=write
    cntk_string="$cn_gpu configFile=$config_write verbosity=0 DeviceNumber=-1 modelName=$cnmodel labelDim=$labelDim featDim=$featDim action=$action ExpDir=$smbrdir"
    local/decode_cntk2.sh --nj $njdec --cmd "$decode_cmd" --acwt $acwt --scoring-opts "$scoring" \
      $graph_src $dev_src $smbrdir/decode_$(basename $dev_src)_it$iter "$cntk_string" || exit 1;
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} $dev_src \
        $expdir/decode_$(basename $dev_src)_it$iter{,fsh_fg}
  done
fi

exit 0;
