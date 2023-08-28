import os
import pprint
import tempfile
import datetime

from typing import Dict, Text

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

from retrieval_model import SimpleRetrievalModel


# https://www.tensorflow.org/api_docs/python/tf/data/Dataset

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

# for retrieval, we keep only the user_id, and movie_title fields in the dataset.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])

# In a real-life setting this should be a temporal split
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

model = SimpleRetrievalModel(
    unique_movie_titles,
    unique_user_ids,
    item_id_key="movie_title",
    user_id_key="user_id",
    embedding_dimension=16,
    candidates=movies
)

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# cache training and test sets
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# setup tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(cached_train, epochs=3, callbacks=[tensorboard_callback])

results = model.evaluate(cached_test, return_dict=True)
print(results)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.item_model)))
)

# Get sample recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")

  # Save the index.
  tf.saved_model.save(index, path)

  # Load it back; can also be done in TensorFlow Serving.
  # loaded = tf.saved_model.load(path)

  # Pass a user id in, get top predicted movie titles back.
  # scores, titles = loaded(["42"])
  # print(f"Recommendations: {titles[0][:3]}")
