# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import argparse
from tensorflow_asr.utils import setup_environment, setup_tpu

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Conformer Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--max_ckpts", type=int, default=10, help="Max number of checkpoints to keep")

parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords")

parser.add_argument("--tfrecords_shards", type=int, default=16, help="Number of tfrecords shards")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

parser.add_argument("--tbs", type=int, default=None, help="Train batch size per replica")

parser.add_argument("--ebs", type=int, default=None, help="Evaluation batch size per replica")

parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

parser.add_argument("--cache", default=False, action="store_true", help="Enable caching for dataset")

parser.add_argument("--subwords", type=str, default=None, help="Path to file that stores generated subwords")

parser.add_argument("--subwords_corpus", nargs="*", type=str, default=[],
                    help="Transcript files for generating subwords")

parser.add_argument("--bfs", type=int, default=100, help="Buffer size for shuffling")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

strategy = setup_tpu()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.keras.asr_dataset_tpu import ASRTFRecordDatasetKerasTPU, get_max_len

from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.models.keras.conformer import Conformer
from tensorflow_asr.optimizers.schedules import TransformerSchedule

config = Config(args.config)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

text_featurizer = CharFeaturizer(config.decoder_config)

files_list = [
    config.learning_config.dataset_config.train_paths,
    config.learning_config.dataset_config.eval_paths
]

max_input_len, max_label_len, max_prediction_len = \
    get_max_len(os.path.join(config.learning_config.dataset_config.tfrecords_dir, "max_len.json"), text_featurizer)

train_dataset = ASRTFRecordDatasetKerasTPU(
    data_paths=config.learning_config.dataset_config.train_paths,
    tfrecords_dir=config.learning_config.dataset_config.tfrecords_dir,
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    augmentations=config.learning_config.augmentations,
    tfrecords_shards=args.tfrecords_shards,
    stage="train",
    max_input_len=max_input_len,
    max_label_len=max_label_len,
    max_prediction_len=max_prediction_len,
    cache=args.cache,
    shuffle=True,
    buffer_size=args.bfs
)

eval_dataset = ASRTFRecordDatasetKerasTPU(
    data_paths=config.learning_config.dataset_config.eval_paths,
    tfrecords_dir=config.learning_config.dataset_config.tfrecords_dir,
    tfrecords_shards=args.tfrecords_shards,
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    stage="eval",
    max_input_len=max_input_len,
    max_label_len=max_label_len,
    max_prediction_len=max_prediction_len,
    cache=args.cache,
    shuffle=True,
    buffer_size=args.bfs
)

with strategy.scope():
    global_batch_size = config.learning_config.running_config.batch_size
    global_batch_size *= strategy.num_replicas_in_sync
    # build model
    conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    input_shape = speech_featurizer.shape
    input_shape[0] = max_input_len

    conformer._build(input_shape)
    conformer.summary(line_length=120)
    optimizer = tf.keras.optimizers.Adam(
        TransformerSchedule(
            d_model=conformer.dmodel,
            warmup_steps=config.learning_config.optimizer_config["warmup_steps"],
            max_lr=(0.05 / math.sqrt(conformer.dmodel))
        ),
        beta_1=config.learning_config.optimizer_config["beta1"],
        beta_2=config.learning_config.optimizer_config["beta2"],
        epsilon=config.learning_config.optimizer_config["epsilon"]
    )

    conformer.compile(optimizer=optimizer, global_batch_size=global_batch_size, blank=text_featurizer.blank)

    train_data_loader = train_dataset.create(global_batch_size)
    eval_data_loader = eval_dataset.create(global_batch_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
        tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
        tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard)
    ]

    conformer.fit(
        train_data_loader,
        epochs=config.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=train_dataset.total_steps
    )
