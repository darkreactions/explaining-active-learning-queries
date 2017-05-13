from __future__ import print_function
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import sys
import unittest

EPS_RATIO = 0.005 # 0.5 %

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def pretty_float(f):
  return "{:.2f}".format(f)

def write(file_name, content):
  with open(file_name, 'w') as f:
    f.write(content.encode('utf8'))

def write_data(file_name, headers, rows):
  with open(file_name,'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)

def load_data(file_name):
  with open(file_name, 'r') as f:
    labels = f.readline().split(',')
    class_index = len(labels) - 1 # assumes that the class is the last column
    data = np.loadtxt(f, delimiter=',')
    X = data[:,:class_index]
    y = data[:, class_index]
    return X, y

def get_feature_names(file_name):
  with open(file_name, 'r') as f:
    labels = f.readline().split(',')
    feature_names = labels[:len(labels)-1]
    assert(len(labels) == len(feature_names) + 1)
    return feature_names

def scale_training_and_test_set(X_train, X_test):
  scaler = StandardScaler().fit(X_train)
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  return X_train_scaled, X_test_scaled

def parse_discretized_feature(discretized_feature):
  """Parse discretized feature provided by lime

  Args:
    discretized_feature: string provided by lime

  Returns:
    tuple with the lower and upper bound
  """
  tokens = discretized_feature.split(" ")
  num_tokens = len(tokens)

  if num_tokens == 5:
    #  n < X <= m
    lower, upper = (float(tokens[0]), float(tokens[-1]))
  elif num_tokens == 3 and tokens[1] == "<=":
    lower, upper = (None, float(tokens[-1]))
  elif num_tokens == 3 and tokens[1] == ">":
    lower, upper = (float(tokens[-1]), None)
  else:
    print("parse_discrete_feature: Invalid argument given")
    raise Exception

  if lower is not None and lower != 0:
    lower = lower * (1 - EPS_RATIO)
  if upper is not None and upper != 1:
    upper = upper * (1 + EPS_RATIO)

  return (lower, upper)

def within_range(value, lower, upper):
  if lower is None:
    return value <= upper
  elif upper is None:
    return lower < value
  else:
    return lower < value <= upper

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

class TestUtil(unittest.TestCase):
  def test_parse_discretized_feature(self):
    f1 = "145.03 < Inorg_drpInorgAtomAtomicRadius <= 171.00"
    f2 = "Inorg_drpInorgAtom_boolean_valence_6_DRP_1_5_any_DRP_1_5 > 0.00"
    f3 = "1440.00 < reaction_time_manual_0 <= 2880.00"
    f4 = "X1 <= 0.73"

    lo1, up1 = parse_discretized_feature(f1)
    lo2, up2 = parse_discretized_feature(f2)
    lo3, up3 = parse_discretized_feature(f3)
    lo4, up4 = parse_discretized_feature(f4)
    self.assertEqual(lo1, 145.03 * (1-EPS_RATIO))
    self.assertEqual(up1, 171.00 * (1+EPS_RATIO))

    self.assertEqual(lo2, 0.00)
    self.assertEqual(up2, None)

    self.assertEqual(lo3, 1440.00 * (1-EPS_RATIO))
    self.assertEqual(up3, 2880.00 * (1+EPS_RATIO))

    self.assertEqual(lo4, None)
    self.assertEqual(up4, 0.73 * (1+EPS_RATIO))

  def test_within_range(self):
    self.assertTrue(within_range(1, 0, 1))
    self.assertTrue(within_range(1, 0, 2))
    self.assertTrue(within_range(100, 1, None))
    self.assertTrue(within_range(1, None, 100))

if __name__ == "__main__":
  unittest.main()
