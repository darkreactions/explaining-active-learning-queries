import sys
sys.path.append('..')

from preprocessing.get_data import get_headers, get_data_list_of_dicts
from util import is_number

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ALL_PATH = "data/Supp_Info_All_XXX.csv"
TRAIN_PATH = "data/Supp_Info_Train_XXX.csv"

def remove_XXX(feature):
  if "XXX" in feature:
    return feature[3:]
  else:
    return feature

def get_unit(feature):
  if "mass" in feature:
    return "grams"
  else:
    return ""

class InterpretableDataManager(object):
  def __init__(self):
    self.feature_names = get_headers(TRAIN_PATH)
    self.rows = get_data_list_of_dicts(TRAIN_PATH)
    self.init_dataset()

  def describe_instance(self, instance_id):
    print "============= An example reaction =============="
    if (instance_id < len(self.rows)):
      row = self.rows[instance_id]
      for feature in self.feature_names:
        value = row[feature]
        if is_number(value) and float(value) == -1.0:
          continue
        else:
          feature = remove_XXX(feature)
          unit = get_unit(feature)
          print " * {0:<13}: {1:} {2:}".format(feature, value, unit)

  def describe_instances(self, instance_ids):
    print "========= A potential variation for probing the region =========="
    d = defaultdict(list)

    for instance_id in instance_ids:
      assert(instance_id < len(self.rows))
      row = self.rows[instance_id]
      for feature in self.feature_names:
        value = row[feature]
        if is_number(value):
          value = float(value)
          if float(value) == -1.0:
            continue

        feature = remove_XXX(feature)
        d[feature].append(value)

    for feature in self.feature_names:
      feature = remove_XXX(feature)
      values = d[feature]
      if values and type(values[0]) is float:
        lo, hi = min(values), max(values)
        if lo == hi:
          continue # no variation
        elif lo == 0 and hi == 1:
          print " * {0:<13}: [{1:}, {2:}]".format(feature, "no", "yes")
        else:
          unit = get_unit(feature)
          print " * {0:<13}: [{1:}, {2:}] {3:}".format(feature, lo, hi, unit)
      elif values and type(values[0]) is str:
        values = list(set(values))
        for i in range(0, len(values), 3):
          vals = ", ".join(values[i:i+3])
          if i == 0:
            print " * {0:<13}: {1:}".format(feature, vals)
          else:
            print " * {0:<13}  {1:}".format("", vals)

  def init_dataset(self):
    data = pd.read_csv(ALL_PATH, quotechar='"', skipinitialspace=True)
    data = data.as_matrix()
    categorical = ["XXXinorg1", "XXXinorg2", "XXXinorg3", "XXXorg1",
                   "XXXorg2", "XXXoxlike1"]
    self.categorical_features = map(lambda x: self.feature_names.index(x),
                                    categorical)
    self.categorical_names = {}

    for feature in self.categorical_features:
      le = LabelEncoder()
      le.fit(data[:, feature])
      data[:, feature] = le.transform(data[:, feature])
      self.categorical_names[feature] = le.classes_

    self.encoder = OneHotEncoder(categorical_features=self.categorical_features,
                                 sparse=False)
    data.astype(float)
    self.encoder.fit(data)
    encoded_all = self.encoder.transform(data)
    num_train = len(self.rows)

    self.X_train = data[:num_train]
    self.X_test = data[num_train:]
    self.X_train_encoded = encoded_all[:num_train]
    self.X_test_encoded = encoded_all[num_train:]

if __name__ == "__main__":
  manager = InterpretableDataManager()
  manager.describe_instance(0)
  print ""
  manager.describe_instance(42)
  print ""
  instance_ids = [1216, 1199, 1505, 155, 176, 2450, 2058, 1248, 1001, 333]
  manager.describe_instances(instance_ids)
  print ""
  manager.describe_instance(155)
