import numpy as np
import sklearn.cluster
import unittest
from certainty import filter_uncertain
from util import eprint, parse_discretized_feature, within_range
from random import sample
from scipy.spatial.distance import euclidean

def get_indices_exp_region(exp, data_manager, unlabeled_indices, y_certainty):
  """
  Return indices of instances in an explanatory region
  """

  uncertain_exp_ids = filter(lambda x: x[1] < 0, exp.as_map()[1])
  feature_ids = map(lambda x: data_manager.feature_id_map[x[0]],
                    uncertain_exp_ids)

  uncertain_exp_list = exp.domain_mapper.map_exp_ids(uncertain_exp_ids)
  bounds = map(lambda x: parse_discretized_feature(x[0]), uncertain_exp_list)
  id_bounds = zip(feature_ids, bounds)
  uncertain_indices_pool = filter_uncertain(unlabeled_indices, y_certainty)
  X_train = data_manager.X_train

  indices_exp_region = uncertain_indices_pool
  for feature_id, bound in id_bounds:
    lower, upper = bound
    indices_exp_region = filter(lambda i: within_range(X_train[i][feature_id],
                                                       lower, upper),
                                indices_exp_region)
  return indices_exp_region

def query_least_confident(indices, certainties, exclusion):
  """
  Select the next query.

  Args:
      indices: indices from which to select the next query from
      certainties: certainties of all instances
      exclusion: a set of indices that are not allowed to be selected

  Returns:
      the index of next query
  """
  indices = filter(lambda x: x not in exclusion, indices)
  certainties = map(lambda x: certainties[x], indices)
  min_index = np.argmin(certainties)
  return indices[min_index]

def label_batch(ds, y, batch):
  """
  Label a batch of instance.

  Args:
      ds: Libact dataset
      y: labels
      batch: a set of ids to label
  """
  for ask_id in batch:
    ds.update(ask_id, y[ask_id])

def select_batch(n, indices, instances, certainties, selection_strategy):
  """
  Selects a batch of instances given a selection strategy.

  Args:
      n: the maximum number instances to be selected
      indices: indices of instances in the original dataset
      instances: instances (should be scaled)
      certainties: certainties of the instances
      selection_strategy: "random", "q-best", "k-means-uncertain",
                          "k-means-closest"

  Returns:
      a list of indices of instances selected as a batch.
  """
  if n >= len(indices):
    return indices

  if selection_strategy == "random":
    return sample(indices, n)
  elif selection_strategy == "q-best":
    indices_certainties = zip(indices, certainties)
    indices_certainties.sort(cmp=lambda x,y: cmp(x[1], y[1]))
    return map(lambda x: x[0],indices_certainties[:n])
  elif selection_strategy in ["k-means-uncertain", "k-means-closest"]:
    return select_batch_k_means(n, indices, instances, certainties, selection_strategy)
  else:
     raise Exception("selection_strategy must be 'random', 'q-best'," +
                     " 'k-means-uncertain' or 'k-means-closest'")

def select_batch_k_means(n, indices, instances, certainties, selection_strategy):
  centroids, labels, _ = sklearn.cluster.k_means(instances, n)
  distances = [] # distance to the closest centroid

  if selection_strategy not in ["k-means-uncertain", "k-means-closest"]:
    raise Exception("selection_strategy must be either " +
                    "'k-means-uncertain' or 'k-means-closest'")

  if selection_strategy == "k-means-closest":
    for i, instance in enumerate(instances) :
      closest_centroid = centroids[labels[i]]
      distance = euclidean(instance, closest_centroid)
      distances.append(distance)
    selection_base_values = distances
  else:
    selection_base_values = certainties

  batch_indices = [None] * n
  batch_values = [None] * n

  for i, index in enumerate(indices):
    cluster_id = labels[i]
    current_value = selection_base_values[i]
    batch_value = batch_values[cluster_id]
    if batch_value is None or batch_value > current_value:
      batch_indices[cluster_id] = index
      batch_values[cluster_id] = current_value

  return filter(lambda x: x is not None, batch_indices)

class TestUtil(unittest.TestCase):
  def test_query_least_confident(self):
    indices = [1,2,3,4,5]
    certainties = [0.5, 0.7, 0.1, 0.9, 0.8, 1.0]
    exclusion = set()
    query_id = query_least_confident(indices, certainties, exclusion)
    self.assertEqual(2, query_id)

    exclusion.add(2)
    query_id = query_least_confident(indices, certainties, exclusion)
    self.assertEqual(1, query_id)

    exclusion.add(1)
    query_id = query_least_confident(indices, certainties, exclusion)
    self.assertEqual(4, query_id)


  def test_q_best(self):
    indices = [1,2,3,4]
    instances = [[1,1],[2,2],[3,3], [4,4]]
    certainties = [0.5, 0.7, 0.1, 0.9]
    batch = select_batch(3, indices, instances, certainties, "q-best")
    self.assertEqual([3,1,2], batch)

  def test_k_means_uncertain(self):
    indices = range(6)
    instances = [[1,1],[2,2],[3,3], [10,10], [11,11], [12,12]]
    certainties = [0.1, 0.4, 0.2, 0.2, 0.3, 0.1]
    batch = select_batch(2, indices, instances, certainties, "k-means-uncertain")
    self.assertEqual(set([0, 5]), set(batch))

  def test_k_means_closest(self):
    indices = range(6)
    instances = [[1,1],[2,2],[3,3], [10,10], [11,11], [12,12]]
    certainties = [0.1, 0.4, 0.2, 0.2, 0.3, 0.1]
    batch = select_batch(2, indices, instances, certainties, "k-means-closest")
    self.assertEqual(set([1, 4]), set(batch))

if __name__ == "__main__":
  unittest.main()
