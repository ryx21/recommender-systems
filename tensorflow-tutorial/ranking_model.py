import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs


class SimpleRankingModel(tfrs.models.Model):

    def __init__(self,
    	item_ids: np.array,
		user_ids: np.array,
		item_id_key: str,
		user_id_key: str,
        rating_key: str,
		embedding_dimension: int,
    ):

        self.item_id_key = item_id_key
        self.user_id_key = user_id_key
        self.rating_key = rating_key

        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.item_embeddings = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=item_ids, mask_token=None),
        tf.keras.layers.Embedding(len(item_ids) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

        # Define Ranking task (loss and metrics)
        self.task = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
    def call(self, features: Dict[str, tf.Tensor]):
        user_id, movie_title = features[self.user_id_key], features[self.item_id_key]
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.item_embeddings(movie_title)
        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop(self.rating_key)
        
        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)
