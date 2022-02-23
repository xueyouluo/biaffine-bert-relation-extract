#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2022 Xueyou.Luo
BASED ON Google_BERT.
"""
from __future__ import absolute_import, division, print_function

import collections
import json
import os
import pdb
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import modeling
import optimization
import tokenization
from utils import get_biaffine_predicate

# 这里为了避免打印重复的日志信息
tf.get_logger().propagate = False

flags = tf.flags

FLAGS = flags.FLAGS

## K-fold
flags.DEFINE_integer("fold_id", 0, "which fold")
flags.DEFINE_integer("fold_num", 1, "total fold number")

flags.DEFINE_integer("seed", 20190525, "random seed")

flags.DEFINE_string(
    "task_name", "spo", "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_train_and_eval", False,
    "Whether to run training and evaluation."
)
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer(
    "train_batch_size", 64,
    "Total batch size for training.")

flags.DEFINE_integer(
    "eval_batch_size", 32,
    "Total batch size for eval.")

flags.DEFINE_integer(
    "predict_batch_size", 32,
    "Total batch size for predict.")

flags.DEFINE_float(
    "learning_rate", 5e-6,
    "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 10.0,
    "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
    "How often to save the model checkpoint.")

flags.DEFINE_bool("horovod", False,
                  "Whether to use Horovod for multi-gpu runs")
flags.DEFINE_bool(
    "amp", False, "Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.")
flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")
flags.DEFINE_string(
    "pooling_type", 'last', "last | first_last "
)
# Dropout
flags.DEFINE_float("embedding_dropout", 0.0, "dropout ratio of embedding")
flags.DEFINE_float("spatial_dropout", 0.0,
                   "dropout ratio of embedding, in channel")
flags.DEFINE_float("bert_dropout", 0.0, "dropout ratio of bert")
# FGM
flags.DEFINE_bool(
    "use_fgm", False,
    "Whether to use FGM to train model.")
flags.DEFINE_float("fgm_epsilon", 0.3, "The epsilon value for FGM")
flags.DEFINE_float("fgm_loss_ratio", 1.0, "The ratio of fgm loss")
flags.DEFINE_float("head_lr_ratio", 1.0, "The ratio of header learning rate")
flags.DEFINE_bool("use_bilstm", False,
                  "Whether to use Bi-LSTM in the last layer.")
flags.DEFINE_bool("electra", False, "Whether to use electra")
flags.DEFINE_bool("dp_decode", False, "Whether to use dp to decode")
flags.DEFINE_integer("biaffine_size", 150, "biaffine size")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, raw_text=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.raw_text = raw_text

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, span_mask, gold_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.span_mask = span_mask
        self.gold_labels = gold_labels

    def to_dict(self):
        return {
            "input_ids":self.input_ids,
            "input_mask":self.input_mask,
            "segment_ids":self.segment_ids,
            "span_mask":self.span_mask,
            "gold_labels":self.gold_labels,

        }

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class SPOObject:
    def __init__(self,s,p,o,st,ss,se,ot,os,oe):
        self.s = s
        self.o = o
        self.st = st
        self.p = p
        self.ot = ot
        self.ss = ss
        self.se = se
        self.os = os
        self.oe = oe

    def __str__(self):
        return f"{self.s} | {self.st} - {self.p} - {self.o} | {self.ot}"
        
    @classmethod
    def from_item(cls,text,item):
        s = item['subject']
        p = item['predicate']
        o = item['object']
        st = 'entity' #item['subject_type']
        ot = 'entity' #item['object_type']
        ss = text.find(s)
        if ss == -1:
            return None
        se = ss + len(s) - 1

        os = text.find(o)
        if os == -1:
            return None

        oe = os + len(o) - 1
        return cls(s,p,o,st,ss,se,ot,os,oe)

class SPOProcessor(DataProcessor):
    def __init__(self, fold_id=0, fold_num=0, max_seq_length=128):
        self.fold_id = fold_id
        self.fold_num = fold_num
        self.max_seq_length = max_seq_length

    def get_train_examples(self, data_dir, file_name='train_data.json'):
        examples = []
        
        for i, line in enumerate(open(os.path.join(data_dir, file_name))):
            item = json.loads(line)
            guid = "%s-%s" % ('train', i)
            text = item['text'].strip()
            if len(text) > self.max_seq_length:
                for step in range(0,len(text),self.max_seq_length):
                    _text = text[step:step+self.max_seq_length].strip()
                    if len(_text) < 5:
                        continue
                    label = item['spo_list']
                    label = self.spo_convert(_text, label)
                    examples.append(InputExample(guid=guid, text=_text, label=label))
            else:
                label = item['spo_list']
                label = self.spo_convert(text, label)
                examples.append(InputExample(guid=guid, text=text, label=label))

                
        random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir, file_name="dev_data.json"):
        examples = []
        for i, line in enumerate(open(os.path.join(data_dir, file_name))):
            item = json.loads(line)
            guid = '%s-%s' % ('dev', i)
            text = item['text'].strip()
            if len(text) > self.max_seq_length:
                for step in range(0,len(text),self.max_seq_length):
                    _text = text[step:step+self.max_seq_length].strip()
                    if len(_text) < 5:
                        continue
                    label = item['spo_list']
                    label = self.spo_convert(_text, label)
                    examples.append(InputExample(guid=guid, text=_text, label=label))
            else:
                label = item['spo_list']
                label = self.spo_convert(text, label)
                examples.append(InputExample(guid=guid, text=text, label=label))
        
        return examples

    def get_test_examples(self, data_dir, file_name="final_test.txt"):
        examples = []
        return examples

    def get_ner_labels(self):
        # labels = ["景点", "作品", "书籍", "歌曲", "气候", "生物", "出版社", "目", "Number", "地点", "网络小说", "历史人物", "网站", "音乐专辑", "图书作品", "城市", "人物", "Text", "学校", "影视作品", "企业", "Date", "学科专业", "语言", "电视综艺", "机构", "行政区", "国家"]
        labels = ['entity']
        return labels

    def get_predicate_labels(self):
        labels = ["祖籍", "父亲", "总部地点", "出生地", "目", "面积", "简称", "上映时间", "妻子", "所属专辑", "注册资本", "首都", "导演", "字", "身高", "出品公司", "修业年限", "出生日期", "制片人", "母亲", "编剧", "国籍", "海拔", "连载网站", "丈夫", "朝代", "民族", "号", "出版社", "主持人", "专业代码", "歌手", "作词", "主角", "董事长", "成立日期", "毕业院校", "占地面积", "官方语言", "邮政编码", "人口数量", "所在城市", "作者", "作曲", "气候", "嘉宾", "主演", "改编自", "创始人"]
        return labels

    def get_all_labels(self):
        link_types = {
            "SH2OH", # subject head to object head
            "OH2SH", # object head to subject head
            "ST2OT", # subject tail to object tail
            "OT2ST", # object tail to subject tail
        }
        tags = {''.join([ent, "EH2ET"]) for ent in self.get_ner_labels()} # EH2ET: entity head to entity tail
        tags |= {''.join([rel, lt]) for rel in self.get_predicate_labels() for lt in link_types}

        return sorted(tags)

    def spo_convert(self, text, label):
        new_labels = []
        for x in label:
            spo = SPOObject.from_item(text,x)
            if spo:
                new_labels.append(spo)
        return new_labels

def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, is_training):
    tokens = tokenizer.tokenize(example.text)
    text = example.text
    if len(tokens) > max_seq_length:
        tokens = tokens[0:max_seq_length]
        text = text[0:max_seq_length]
    try:
        assert len(text) == len(tokens)
    except:
        print(text)
        print(tokens)
        print(example.guid)
        raise 

    ntokens = []
    segment_ids = []
    span_mask = []

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        span_mask.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        span_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(span_mask) == max_seq_length
    
    if is_training:
        size = len(text)
        n = size * (size + 1) // 2
        gold_labels = [[0] * len(label_map) for _ in range(n)]

        def get_position(s,e):
            return s * size + e - s * (s + 1) // 2

        for spo in example.label:
            if spo.ss >= size or spo.se >= size or spo.os >= size or spo.oe >= size:
                continue
            gold_labels[get_position(spo.ss,spo.se)][label_map[''.join([spo.st,'EH2ET'])]] = 1
            gold_labels[get_position(spo.os,spo.oe)][label_map[''.join([spo.ot,'EH2ET'])]] = 1
            if spo.ss > spo.os:
                gold_labels[get_position(spo.os,spo.ss)][label_map[''.join([spo.p,'OH2SH'])]] = 1
            else:
                gold_labels[get_position(spo.ss,spo.os)][label_map[''.join([spo.p,'SH2OH'])]] = 1

            if spo.se > spo.oe:
                gold_labels[get_position(spo.oe,spo.se)][label_map[''.join([spo.p,'OT2ST'])]] = 1
            else:
                gold_labels[get_position(spo.se,spo.oe)][label_map[''.join([spo.p,'ST2OT'])]] = 1

    else:
        gold_labels = [[0] * len(label_map)]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        span_mask=span_mask,
        gold_labels=gold_labels,
    )
    return feature

def generator_based_input_fn_builder(examples, label_list, max_seq_length, tokenizer, is_training, batch_size):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    def generator():
        for (ex_index, example) in enumerate(examples):
            feature = convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer,
                                            is_training)
            yield feature.to_dict()

    def input_fn(params):
        d = tf.data.Dataset.from_generator(
            generator,
            output_types={
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                'segment_ids': tf.int32,
                'span_mask': tf.int32,
                'gold_labels': tf.int32
            },
            output_shapes={
                "input_ids": tf.TensorShape([max_seq_length]),
                "input_mask": tf.TensorShape([max_seq_length]),
                'segment_ids': tf.TensorShape([max_seq_length]),
                'span_mask': tf.TensorShape([max_seq_length]),
                'gold_labels': tf.TensorShape([None,len(label_map)])
            }
        )

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.padded_batch(
            batch_size,
            padded_shapes={
                "input_ids": (tf.TensorShape([max_seq_length])),
                "input_mask": tf.TensorShape([max_seq_length]),
                "segment_ids": tf.TensorShape([max_seq_length]),
                "span_mask": tf.TensorShape([max_seq_length]),
                "gold_labels": tf.TensorShape([None,len(label_map)])
            },
            padding_values={
                'input_ids': 0,
                "input_mask": 0,
                "segment_ids": 0,
                'span_mask': 0,
                'gold_labels': -1 # -1是为了boolen_mask方便而设置
            },
            drop_remainder=False
        ).prefetch(20)
        return d
    return input_fn


def biaffine_mapping(vector_set_1,
                     vector_set_2,
                     output_size,
                     add_bias_1=True,
                     add_bias_2=True,
                     initializer=None,
                     name='Bilinear'):
    """Bilinear mapping: maps two vector spaces to a third vector space.
    The input vector spaces are two 3d matrices: batch size x bucket size x values
    A typical application of the function is to compute a square matrix
    representing a dependency tree. The output is for each bucket a square
    matrix of the form [bucket size, output size, bucket size]. If the output size
    is set to 1 then results is [bucket size, 1, bucket size] equivalent to
    a square matrix where the bucket for instance represent the tokens on
    the x-axis and y-axis. In this way represent the adjacency matrix of a
    dependency graph (see https://arxiv.org/abs/1611.01734).
    Args:
       vector_set_1: vectors of space one
       vector_set_2: vectors of space two
       output_size: number of output labels (e.g. edge labels)
       add_bias_1: Whether to add a bias for input one
       add_bias_2: Whether to add a bias for input two
       initializer: Initializer for the bilinear weight map
    Returns:
      Output vector space as 4d matrix:
      batch size x bucket size x output size x bucket size
      The output could represent an unlabeled dependency tree when
      the output size is 1 or a labeled tree otherwise.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Dynamic shape info
        batch_size = tf.shape(vector_set_1)[0]
        bucket_size = tf.shape(vector_set_1)[1]

        if add_bias_1:
            vector_set_1 = tf.concat(
                [vector_set_1, tf.ones([batch_size, bucket_size, 1])], axis=2)
        if add_bias_2:
            vector_set_2 = tf.concat(
                [vector_set_2, tf.ones([batch_size, bucket_size, 1])], axis=2)

        # Static shape info
        vector_set_1_size = vector_set_1.get_shape().as_list()[-1]
        vector_set_2_size = vector_set_2.get_shape().as_list()[-1]

        if not initializer:
            initializer = tf.orthogonal_initializer()

        # Mapping matrix
        bilinear_map = tf.get_variable(
            'bilinear_map', [vector_set_1_size,
                             output_size, vector_set_2_size],
            initializer=initializer)

        # The matrix operations and reshapings for bilinear mapping.
        # b: batch size (batch of buckets)
        # v1, v2: values (size of vectors)
        # n: tokens (size of bucket)
        # r: labels (output size), e.g. 1 if unlabeled or number of edge labels.

        # [b, n, v1] -> [b*n, v1]
        vector_set_1 = tf.reshape(vector_set_1, [-1, vector_set_1_size])

        # [v1, r, v2] -> [v1, r*v2]
        bilinear_map = tf.reshape(bilinear_map, [vector_set_1_size, -1])

        # [b*n, v1] x [v1, r*v2] -> [b*n, r*v2]
        bilinear_mapping = tf.matmul(vector_set_1, bilinear_map)

        # [b*n, r*v2] -> [b, n*r, v2]
        bilinear_mapping = tf.reshape(
            bilinear_mapping,
            [batch_size, bucket_size * output_size, vector_set_2_size])

        # [b, n*r, v2] x [b, n, v2]T -> [b, n*r, n]
        bilinear_mapping = tf.matmul(
            bilinear_mapping, vector_set_2, adjoint_b=True)

        # [b, n*r, n] -> [b, n, r, n]
        bilinear_mapping = tf.reshape(
            bilinear_mapping, [batch_size, bucket_size, output_size, bucket_size])
        return bilinear_mapping


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, span_mask, num_labels,use_fgm=False, 
                 perturbation=None, spatial_dropout=None,embedding_dropout=0.0,
                 bilstm=None,biaffine_size=150,pooling_type='last'):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False,
        use_fgm=use_fgm,
        perturbation=perturbation,
        spatial_dropout=spatial_dropout,
        embedding_dropout=embedding_dropout
    )

    output_layer = model.get_sequence_output()

    if pooling_type != 'last':
        raise NotImplementedError('没实现。')

    batch_size, seq_length, hidden_size = modeling.get_shape_list(
        output_layer, expected_rank=3)

    if bilstm is not None and len(bilstm) == 2:
        tf.logging.info('Using Bi-LSTM')
        sequence_length = tf.reduce_sum(input_mask, axis=-1)
        with tf.variable_scope('bilstm', reuse=tf.AUTO_REUSE):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=bilstm[0],
                cell_bw=bilstm[1],
                dtype=tf.float32,
                sequence_length=sequence_length,
                inputs=output_layer
            )
            output_layer = tf.concat(outputs, -1)

    if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    # Magic Number
    size = biaffine_size
    
    starts = tf.layers.dense(output_layer, size, kernel_initializer=tf.truncated_normal_initializer(
        stddev=0.02), name='start', reuse=tf.AUTO_REUSE)
    ends = tf.layers.dense(output_layer, size, kernel_initializer=tf.truncated_normal_initializer(
        stddev=0.02), name='end', reuse=tf.AUTO_REUSE)

    biaffine = biaffine_mapping(
        starts,
        ends,
        num_labels,
        add_bias_1=True,
        add_bias_2=True,
        initializer=tf.zeros_initializer(),
        name='biaffine')

    # [B,1,L] [B,L,1] -> [B,L,L]
    span_mask = tf.cast(span_mask, dtype=tf.bool)
    candidate_scores_mask = tf.logical_and(tf.expand_dims(
        span_mask, axis=1), tf.expand_dims(span_mask, axis=2))
    # B,L,L
    sentence_ends_leq_starts = tf.tile(
        tf.expand_dims(
            tf.logical_not(tf.sequence_mask(tf.range(seq_length), seq_length)),
            0),
        [batch_size, 1, 1]
    )
    # B,L,L
    candidate_scores_mask = tf.logical_and(
        candidate_scores_mask, sentence_ends_leq_starts)
    # B*L*L
    flattened_candidate_scores_mask = tf.reshape(candidate_scores_mask, [-1])

    def get_valid_scores(biaffine):
        # B,L,L,N
        candidate_scores = tf.transpose(biaffine, [0, 1, 3, 2])
        candidate_scores = tf.boolean_mask(tf.reshape(
            candidate_scores, [-1, num_labels]), flattened_candidate_scores_mask)
        return candidate_scores

    # 只获取合法位置的logits，最终变成[X,num_labels]，X大小与batch中数据相关
    candidate_ner_scores = get_valid_scores(biaffine)

    return candidate_ner_scores, model

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    参考苏神的实现：https://github.com/bojone/bert4keras/blob/3161648d20bfe7f501297d4bb33a0bad1ffd4002/bert4keras/backend.py#L250
    y_pred: (batch_size, shaking_seq_len, type_size)
    y_true: (batch_size, shaking_seq_len, type_size)
    y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
            1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
    """
    y_true = tf.cast(y_true,y_pred.dtype)
    y_pred = (1 - 2 * y_true) * y_pred
    y_neg = y_pred - y_true * 1e20
    y_pos = y_pred - (1 - y_true) * 1e20
    zeros = tf.zeros_like(y_pred[..., :1])
    y_neg = tf.concat([y_neg, zeros], axis=-1)
    y_pos = tf.concat([y_pos, zeros], axis=-1)
    neg_loss = tf.reduce_logsumexp(y_neg, axis=-1)
    pos_loss = tf.reduce_logsumexp(y_pos, axis=-1)
    return neg_loss + pos_loss

def model_fn_builder(bert_config, num_labels, init_checkpoint=None, learning_rate=None,
                     num_train_steps=None, num_warmup_steps=None,
                     use_one_hot_embeddings=False, hvd=None, amp=False):
    def model_fn(features, labels, mode, params):
        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info(
                "  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        span_mask = features["span_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if is_training and FLAGS.bert_dropout > 0.0:
            bert_config.hidden_dropout_prob = FLAGS.bert_dropout
            bert_config.attention_probs_dropout_prob = FLAGS.bert_dropout

        batch_size = tf.shape(input_ids)[0]
        spatial_dropout_layer = None
        if is_training and FLAGS.spatial_dropout > 0.0:
            spatial_dropout_layer = tf.keras.layers.SpatialDropout1D(
                FLAGS.spatial_dropout)

        bilstm = None
        if FLAGS.use_bilstm:
            fw_cell = tf.nn.rnn_cell.LSTMCell(bert_config.hidden_size)
            bw_cell = tf.nn.rnn_cell.LSTMCell(bert_config.hidden_size)
            if is_training:
                fw_cell = lstm_dropout_warpper(fw_cell)
                bw_cell = lstm_dropout_warpper(bw_cell)
            bilstm = (fw_cell, bw_cell)

        reuse_model = FLAGS.use_fgm
        candidate_ner_scores, model = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, span_mask, num_labels,
            spatial_dropout=spatial_dropout_layer, bilstm=bilstm, use_fgm=reuse_model,
            biaffine_size=FLAGS.biaffine_size,pooling_type=FLAGS.pooling_type,embedding_dropout=FLAGS.embedding_dropout
            )

        output_spec = None
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            if init_checkpoint and (hvd is None or hvd.rank() == 0):
                (assignment_map,
                 initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, convert_electra=FLAGS.electra)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.compat.v1.logging.info("**** Trainable Variables ****")

            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                          init_string)

            gold_labels = features['gold_labels']
            gold_labels = tf.reshape(gold_labels,[-1,num_labels])
            # 根据-1的padding来获取真实的label，得到[X,num_labels]，X与candidate_ner_scores一致
            gold_labels = tf.boolean_mask(gold_labels,tf.not_equal(gold_labels[...,0],-1))
            total_loss = multilabel_categorical_crossentropy(candidate_ner_scores,gold_labels)
            total_loss = tf.reduce_sum(total_loss) / tf.to_float(batch_size)
            # 只计算有label的位置的准确率，避免大量0的干扰
            acc = tf.metrics.accuracy(gold_labels,tf.cast(tf.greater(candidate_ner_scores,0),gold_labels.dtype),weights=tf.cast(tf.greater(gold_labels,0),gold_labels.dtype))

            tensor_to_log = {
                "accuracy": acc[1] * 100
            }
            if FLAGS.use_fgm:
                embedding_output = model.get_embedding_output()
                grad, = tf.gradients(
                    total_loss,
                    embedding_output,
                    aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
                grad = tf.stop_gradient(grad)
                perturbation = modeling.scale_l2(grad, FLAGS.fgm_epsilon)
                adv_candidate_ner_scores, _ = create_model(
                    bert_config, is_training, input_ids, input_mask, segment_ids, span_mask, num_labels,
                    use_fgm=True, perturbation=perturbation, spatial_dropout=spatial_dropout_layer, bilstm=bilstm,
                    biaffine_size=FLAGS.biaffine_size,pooling_type=FLAGS.pooling_type,embedding_dropout=FLAGS.embedding_dropout
                    )

                adv_loss = multilabel_categorical_crossentropy(adv_candidate_ner_scores,gold_labels)
                adv_loss = tf.reduce_sum(adv_loss) / tf.to_float(batch_size)

                total_loss = (total_loss + FLAGS.fgm_loss_ratio *
                              adv_loss) / (1 + FLAGS.fgm_loss_ratio)

            train_op, _ = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, hvd, amp, head_lr_ratio=FLAGS.head_lr_ratio)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[tf.train.LoggingTensorHook(tensor_to_log, every_n_iter=50)])
        elif mode == tf.estimator.ModeKeys.EVAL:
            # Fake metric
            def metric_fn():
                unused_mean = tf.metrics.mean(tf.ones([2, 3]))
                return {
                    "unused_mean": unused_mean
                }
            eval_metric_ops = metric_fn()
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=tf.constant(1.0),
                eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "score": tf.expand_dims(candidate_ner_scores, 0), 
                    'batch_size': tf.expand_dims(batch_size, 0)} 
            )
        return output_spec

    return model_fn


def main(_):
    # Set different seed for different model
    seed = FLAGS.seed + FLAGS.fold_id
    tf.random.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if FLAGS.horovod:
        import horovod.tensorflow as hvd
        hvd.init()

    processors = {
        "spo": SPOProcessor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_train_and_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tf.io.gfile.makedirs(FLAGS.output_dir)

    processor = processors[task_name](FLAGS.fold_id,FLAGS.fold_num,FLAGS.max_seq_length)

    label_list = processor.get_all_labels()

    # 避免alignment的处理
    tokenizer = tokenization.SimpleTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    master_process = True
    training_hooks = []
    global_batch_size = FLAGS.train_batch_size
    hvd_rank = 0

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if FLAGS.horovod:
        global_batch_size = FLAGS.train_batch_size * hvd.size()
        master_process = (hvd.rank() == 0)
        hvd_rank = hvd.rank()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        if hvd.size() > 1:
            training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    if FLAGS.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
        if FLAGS.amp:
            tf.enable_resource_variables()

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir if master_process else None,
        session_config=config,
        log_step_count_steps=50,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
        keep_checkpoint_max=1)

    if master_process:
        tf.compat.v1.logging.info("***** Configuaration *****")
        for key in FLAGS.__flags.keys():
            tf.compat.v1.logging.info(
                '  {}: {}'.format(key, getattr(FLAGS, key)))
        tf.compat.v1.logging.info("**************************")

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train or FLAGS.do_train_and_eval:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / global_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        start_index = 0
        end_index = len(train_examples)

        if FLAGS.horovod:
            num_examples_per_rank = len(train_examples) // hvd.size()
            remainder = len(train_examples) % hvd.size()
            if hvd.rank() < remainder:
                start_index = hvd.rank() * (num_examples_per_rank+1)
                end_index = start_index + num_examples_per_rank + 1
            else:
                start_index = hvd.rank() * num_examples_per_rank + remainder
                end_index = start_index + (num_examples_per_rank)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False,
        hvd=None if not FLAGS.horovod else hvd,
        amp=FLAGS.amp)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    if FLAGS.do_train or FLAGS.do_train_and_eval:
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = generator_based_input_fn_builder(
            examples=train_examples[start_index:end_index],
            label_list=label_list,
            max_seq_length=FLAGS.max_seq_length, 
            tokenizer=tokenizer, 
            is_training=True, 
            batch_size=FLAGS.train_batch_size
        )

    if FLAGS.do_predict or FLAGS.do_eval or FLAGS.do_train_and_eval:
        if FLAGS.do_eval or FLAGS.do_train_and_eval:
            predict_examples = processor.get_dev_examples(FLAGS.data_dir)
        else:
            predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_batch_size = FLAGS.predict_batch_size
        tf.compat.v1.logging.info("***** Running prediction*****")
        tf.compat.v1.logging.info("  Num examples = %d", len(predict_examples))
        tf.compat.v1.logging.info("  Batch size = %d", predict_batch_size)

        predict_input_fn = generator_based_input_fn_builder(
            examples=predict_examples,
            label_list=label_list,
            max_seq_length=FLAGS.max_seq_length, 
            tokenizer=tokenizer, 
            is_training=False, 
            batch_size=predict_batch_size
        )

    if FLAGS.do_train_and_eval:
        raise NotImplementedError('没实现，交给你们自己了。')
    else:
        if FLAGS.do_train:
            estimator.train(input_fn=train_input_fn,
                            max_steps=num_train_steps, hooks=training_hooks)

    if FLAGS.do_eval or FLAGS.do_predict:
        if FLAGS.do_predict:
            raise NotImplementedError('没实现，参考eval的逻辑很容易实现。')
        idx = 0
        # TP - 预测对的数量
        # PN - 预测的数量
        # TN - 真实的数量
        TP,PN,TN = 1e-10,1e-10,1e-10
        for i, prediction in enumerate(tqdm(estimator.predict(input_fn=predict_input_fn, yield_single_examples=True), total=len(predict_examples)//predict_batch_size)):
            scores = prediction['score']
            offset = 0
            bz = prediction['batch_size']
            for j in range(bz):
                example = predict_examples[idx]
                text = example.text
                pred_text = example.text[:FLAGS.max_seq_length]
                size = len(pred_text) * (len(pred_text) + 1) // 2
                pred_score = scores[offset:offset+size]
                idx += 1
                offset += size
                ret = get_biaffine_predicate(pred_text,pred_score,label_list,processor.get_predicate_labels())
                truth = set([(spo.s,spo.p,spo.o) for spo in example.label])
                predict = set([(pred_text[s[0]:s[1]+1],p,pred_text[o[0]:o[1]+1]) for s,p,o in ret])
                TP += len(truth & predict)
                TN += len(truth)
                PN += len(predict)
        print(f'precision {TP/PN}, recall {TP/TN}, f1 {2*TP/(PN+TN)}')

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
