# Copyright 2020 wws (@trillionmonster)
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
import multiprocessing
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
from .asr_dataset import ASRTFRecordDatasetKeras
from ..asr_dataset import AUTOTUNE, TFRECORD_SHARDS
from ..base_dataset import BUFFER_SIZE
from ...featurizers.speech_featurizers import SpeechFeaturizer, read_raw_audio
from ...featurizers.text_featurizers import TextFeaturizer
from ...utils.utils import get_num_batches, float_feature, int64_feature

from ...augmentations.augments import Augmentation


def get_max_len(cache_path,
                text_featurizer: TextFeaturizer,
                source_file_list=None):
    max_input_len = 0
    max_label_len = 0
    max_prediction_len = 0

    if tf.io.gfile.exists(cache_path):
        with tf.io.gfile.GFile(cache_path, mode='r') as gf:
            obj = json.load(gf)
            max_input_len = int(obj["max_input_len"])
            max_label_len = int(obj["max_label_len"])
            max_prediction_len = int(obj["max_prediction_len"])
        return max_input_len, max_label_len, max_prediction_len

    assert source_file_list, cache_path + " not exist source_file_list can not be null"

    for source_file in source_file_list:

        with tf.io.gfile.GFile(source_file, "r") as f:
            # print(source_file)
            lines = f.read().splitlines()
        lines = lines[1:]
        lines = [line.split("\t", 2) for line in lines]
        lines = np.array(lines)

        for line in tqdm(lines, desc="[compute max len ]"):
            duration, text = line[1], line[2]
            audio_feature_len = int(float(duration) * 100 // 1) + 1
            txt_len = len(text_featurizer.extract(text))
            predict_len = txt_len + 1

            if audio_feature_len > max_input_len:
                max_input_len = audio_feature_len
            if txt_len > max_label_len:
                max_label_len = txt_len
            if predict_len > max_prediction_len:
                max_prediction_len = predict_len

        print("max_input_len", max_input_len)
        print("max_label_len", max_label_len)
        print("max_prediction_len", max_prediction_len)

    with tf.io.gfile.GFile(cache_path, mode='w') as gf:
        json.dump(
            {
                "max_input_len": max_input_len,
                "max_label_len": max_label_len,
                "max_prediction_len": max_prediction_len
            }, gf)

    return max_input_len, max_label_len, max_prediction_len


def write_tfrecord_features(shard_path, audio_token_id_pairs):
    with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
        for audio_16k_normal, token_ids in tqdm(audio_token_id_pairs):
            feature = {
                "audio_16k_normal": float_feature(audio_16k_normal),
                "token_ids": int64_feature(token_ids)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            out.write(example.SerializeToString())
    print(f"\nCreated {shard_path}")


class ASRTFRecordDatasetKerasTPU(ASRTFRecordDatasetKeras):
    """ Keras Dataset for ASR using TFRecords """

    def __init__(self,
                 data_paths: list,
                 tfrecords_dir: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 stage: str,
                 max_input_len: int,
                 max_label_len: int,
                 max_prediction_len: int,
                 augmentations: Augmentation = Augmentation(None),
                 tfrecords_shards: int = TFRECORD_SHARDS,
                 cache: bool = False,
                 shuffle: bool = False,

                 buffer_size: int = BUFFER_SIZE):
        super(ASRTFRecordDatasetKerasTPU, self).__init__(
            stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, tfrecords_dir=tfrecords_dir, augmentations=augmentations, cache=cache,
            shuffle=shuffle, buffer_size=buffer_size
        )
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0: raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards
        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)
        self.max_input_len = max_input_len
        self.max_label_len = max_label_len
        self.max_prediction_len = max_prediction_len

    def read_features(self, entries):
        audio_token_id_pairs = []

        for audio, _, transcript in tqdm(entries, desc="reading entries"):
            signal = read_raw_audio(audio, self.speech_featurizer.sample_rate)

            label = self.text_featurizer.extract(transcript)

            audio_token_id_pairs.append([signal, label])

        return audio_token_id_pairs

    def create_tfrecords(self):

        pattern = self.tfrecords_dir + f"{self.stage}*.tfrecord"

        file_list = tf.io.gfile.glob(pattern)

        if len(file_list) == self.tfrecords_shards:
            print(f"TFRecords're already existed: {self.stage}")
            return True

        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)

        print(f"Creating {self.stage}.tfrecord ...")

        entries = self.read_entries()
        if len(entries) <= 0:
            return False

        audio_token_id_pairs = self.read_features(entries)

        def get_shard_path(shard_id):

            return os.path.join(self.tfrecords_dir, f"{self.stage}_{shard_id}.tfrecord")

        shards = [get_shard_path(idx) for idx in range(1, self.tfrecords_shards + 1)]

        splitted_audio_token_id_pairs = np.array_split(audio_token_id_pairs, self.tfrecords_shards)
        with multiprocessing.Pool(self.tfrecords_shards) as pool:
            pool.map(write_tfrecord_features, zip(shards, splitted_audio_token_id_pairs))

        return True

    @tf.function
    def parse(self, record):
        feature_description = {
            "audio_16k_normal": tf.io.FixedLenSequenceFeature([], tf.float32),
            "token_ids": tf.io.FixedLenSequenceFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(record, feature_description)
        audio_16k_normal = example["audio_16k_normal"]
        token_ids = example["token_ids"]

        signal = self.augmentations.before.augment(audio_16k_normal)

        features = self.speech_featurizer.extract(signal)

        features = self.augmentations.after.augment(features)

        features = tf.convert_to_tensor(features, tf.float32)

        prediction = self.text_featurizer.prepand_blank(token_ids)

        prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)

        input_length = tf.cast(tf.shape(features)[0], tf.int32)

        label_length = tf.cast(tf.shape(token_ids)[0], tf.int32)

        token_ids = tf.convert_to_tensor(token_ids, tf.int32)
        return (
            {
                "input": features,
                "input_length": input_length,
                "prediction": prediction,
                "prediction_length": prediction_length
            },
            {
                "label": token_ids,
                "label_length": label_length
            }
        )

    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        # PADDED BATCH the dataset
        input_shape = self.speech_featurizer.shape
        input_shape[0] = self.max_input_len

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                {
                    "input": tf.TensorShape(input_shape),
                    "input_length": tf.TensorShape([]),
                    "prediction": tf.TensorShape([self.max_prediction_len]),
                    "prediction_length": tf.TensorShape([])
                },
                {
                    "label": tf.TensorShape([self.max_label_len]),
                    "label_length": tf.TensorShape([])
                },
            ),
            padding_values=(
                {
                    "input": 0.,
                    "input_length": 0,
                    "prediction": self.text_featurizer.blank,
                    "prediction_length": 0
                },
                {
                    "label": self.text_featurizer.blank,
                    "label_length": 0
                }
            ),
            drop_remainder=True
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        self.total_steps = get_num_batches(self.total_steps, batch_size)
        return dataset
