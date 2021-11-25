#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import random
import collections
import os
import pdb
import pickle
from tqdm import tqdm

def write_instance_to_example_files(train_path, output_dir, max_seq_length=128, stride=128, num_output=10):
    """Create TF example files from `TrainingInstance`s."""

    def to_tfrecord(input_list, writers):
        writer_index = 0
        total_written = 0
        for article in tqdm(input_list):
            samples = []
            start_point = 0
            while start_point < len(article) - max_seq_length:
                samples.append(article[start_point: start_point + max_seq_length])
                start_point += stride
            if start_point < len(article):
                samples.append(article[len(article) - max_seq_length:])
            random.shuffle(samples)
            for (inst_index, input_ids) in enumerate(samples):
                features = collections.OrderedDict()
                features["input_ids"] = create_int_feature(input_ids)

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writers[writer_index].write(tf_example.SerializeToString())
                if total_written % 100 == 0:
                    #update writer_index
                    writer_index = (writer_index + 1) % len(writers)
                total_written += 1
        for writer in writers:
            writer.close()
        return total_written

    writers = []
    output_files = [os.path.join(output_dir, 'data_{}_.tfrecord'.format(i)) for i in range(num_output)]
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)
    
    total_written = to_tfrecord(input_list, writers)
    tf.compat.v1.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

if __name__ == "__main__":
   write_instance_to_example_files(train_path='wikicorpus_en_one_article_per_line.pkl', output_dir='./tfrecords/')