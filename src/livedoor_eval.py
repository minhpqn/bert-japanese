from argparse import ArgumentParser
import sys
import pathlib
from sklearn import metrics

import tokenization_sentencepiece as tokenization
from run_classifier import LivedoorProcessor
from run_classifier import model_fn_builder
from run_classifier import file_based_input_fn_builder
from run_classifier import file_based_convert_examples_to_features
from utils import str_to_value

sys.path.append("..")

from bert import modeling
import tensorflow as tf

import json
import tempfile

import configparser
import glob
import os


class FLAGS(object):
    '''Parameters.'''

    def __init__(self,
                 data_dir,
                 output_dir,
                 bert_model_dir,
                 fine_tune_dir,
                 do_lower_case=True,
                 use_tpu=False,
                 max_seq_length=512,
                 predict_batch_size = 4
                 ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_file = os.path.join(bert_model_dir, "wiki-ja.model")
        self.vocab_file = os.path.join(bert_model_dir, "wiki-ja.vocab")
        output_ckpts = glob.glob("{}/model.ckpt*data*".format(fine_tune_dir))
        latest_ckpt = sorted(output_ckpts)[-1]
        self.init_checkpoint = latest_ckpt.split('.data-00000-of-00001')[0]

        self.do_lower_case = do_lower_case
        self.use_tpu = use_tpu
        self.max_seq_length = max_seq_length
        self.predict_batch_size = predict_batch_size

        # The following parameters are not used in predictions.
        # Just use to create RunConfig.
        self.master = None
        self.save_checkpoints_steps = 1
        self.iterations_per_loop = 1
        self.num_tpu_cores = 1
        self.learning_rate = 0
        self.num_warmup_steps = 0
        self.num_train_steps = 0
        self.train_batch_size = 0
        self.eval_batch_size = 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-keep_case", action="store_true", help="Keep the case (do not do lower case)")
    parser.add_argument("-use_tpu", action="store_true", help="Use TPU")
    parser.add_argument("-max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("-data_dir", required=True, help="Data directory")
    parser.add_argument("-output_dir", required=True, help="Output directory")
    parser.add_argument("-bert_model_dir", required=True, help="Path to pretrained bert model dir")
    parser.add_argument("-fine_tune_dir", required=True, help="Path to output fine-tuned dir")
    args = parser.parse_args()

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    CURDIR = os.getcwd()
    CONFIGPATH = os.path.join(CURDIR, os.pardir, 'config.ini')
    config = configparser.ConfigParser()
    config.read(CONFIGPATH)

    bert_config_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.json')
    bert_config_file.write(json.dumps({k: str_to_value(v) for k, v in config['BERT-CONFIG'].items()}))
    bert_config_file.seek(0)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file.name)

    FLAGS = FLAGS(args.data_dir, args.output_dir,
                  args.bert_model_dir, args.fine_tune_dir,
                  do_lower_case=True, use_tpu=args.use_tpu,
                  max_seq_length=args.max_seq_length
                  )
    processor = LivedoorProcessor()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        model_file=FLAGS.model_file, vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    print("# Loaded {} test examples.".format(len(predict_examples)))

    predict_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.tf_record')

    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file.name)

    predict_drop_remainder = True if FLAGS.use_tpu else False

    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file.name,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    # It will take a few hours on CPU environment.
    result = list(result)

    true_labels = [e.label for e in predict_examples]
    predict_labels = [ label_list[elem['probabilities'].argmax()] for elem in result ]

    print("Accuracy: {}".format(metrics.accuracy_score(true_labels, predict_labels)))
    print(metrics.classification_report(true_labels, predict_labels))
    print(metrics.confusion_matrix(true_labels, predict_labels))

