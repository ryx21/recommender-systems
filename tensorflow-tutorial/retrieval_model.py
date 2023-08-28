import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

# Based largely on this code: https://www.tensorflow.org/recommenders/examples/basic_retrieval

class SimpleRetrievalModel(tfrs.Model):

  def __init__(self, item_ids: np.array, user_ids: np.array, item_id_key: str, user_id_key: str, embedding_dimension: int, candidates):
    super().__init__()

    self.item_id_key = item_id_key
    self.user_id_key = user_id_key
    
    self.item_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None),
        tf.keras.layers.Embedding(len(item_ids) + 1, embedding_dimension)
    ])

    self.user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dimension)
    ])

    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=candidates.batch(128).map(self.item_model),
            ks=(10,)
        )
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features[self.user_id_key])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.item_model(features[self.item_id_key])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_movie_embeddings)