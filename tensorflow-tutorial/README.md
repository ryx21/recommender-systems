# Tensorflow Recommenders Tutorial

## General Links
* Overview: https://www.tensorflow.org/recommenders
* AWS Services: https://www.youtube.com/watch?v=FDEpdNdFglI&ab_channel=BeABetterDev


## 1. Basic Retrieval

Tutorial: [link](https://www.tensorflow.org/recommenders/examples/basic_retrieval)

What does `tfrs.tasks.Retrieval` ([docs](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval)) do under the hood? .
* Given a batch of `query_embeddings` and `candidate_embeddings` the prediction target is an identity matrix of shape `[num_queries, num_candidates]`. In other words, you want to predict `1` for a positive query-candidate pair and `0` for any non-existant pair. The loss function used by default is simply `tf.keras.losses.CategoricalCrossentropy` ([docs](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy)). The score for each pair is taken as a dot product between the `query` and `candidate`. 

```
>> label
   A  B  C
X [1, 0, 0]
Y [0, 1, 0]
Z [0, 0, 1]

>> score
   A  B  C
X [., ., .]
Y [., ., .]
Z [., ., .]
```

* If multiple queries are paired with the same candidate, for any one of these queries, its row in the target matrix will have accidental negatives. `remove_accidental_hits` can be enabled to zero the corresponding logit in the `score` matrix.
* In the example below, Y-B and Z-B have occurences of accidental negatives
```
>> label
   A  B  B
X [1, 0, 0]
Y [0, 1, 0]
Z [0, 0, 1]

>> score
   A  B  B
X [., ., .]
Y [., ., 0]
Z [., 0, .]
```
* If `num_hard_negatives` hard negatives is enabled, only the negative samples with the biggest score logits are included.

What does `tfrs.metrics.FactorizedTopK` ([docs](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/metrics/FactorizedTopK)) do under the hood?
* By default this calculates top K categorical accuracy - i.e. how often the true candidate is in the top K candidates (across all candidates) for a given query.
* I haven't brought myselft to reading the code in detail, but I think this just calculates this metric streaming fashion over a series of batches covering a full dataset.
* `query_from_exclusions` can be optionally used to remove candidates from the top K calculation - commonly used for removing previously seen candidates


What does `tfrs.layers.factorized_top_k.BruteForce` ([docs](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/layers/factorized_top_k/BruteForce)) do under the hood?
* Provide this with a `query_model` and build a retrieval (candidate) index, so when called it queries the index and returns the top K candidates
* Exchaustive, exact search for nearest neighbours is very computationally intense
* Can use `query_from_exclusions` too
* `tfrs.layers.factorized_top_k.ScaNN` ([docs](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/layers/factorized_top_k/ScaNN)) is a more efficient approximate alternative better for large scale retrieval (large number of candidates).
