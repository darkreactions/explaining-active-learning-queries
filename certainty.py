import unittest, re
import numpy as np
from util import write, parse_discretized_feature, is_number, within_range
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, matthews_corrcoef

UNCERTAIN, CERTAIN = range(2)
NUM_EXPLATIONS = 5
DISCRETIZER = "entropy"

def get_certainty_labels(model, X_all, X_pool, percentile):
  """
  returns certainty labels of |X_all| given |model|
  certainty
    | prediction >= |threshold| = 1
    | prediction <  |threshold| = 0
    where threshold is set based on |X_pool| and |percentile|
  """
  threshold = get_certainty_threshold(model, X_pool, percentile)
  certainties_all = get_certainties(model, X_all)
  return discretize_certainties(certainties_all, threshold)

def get_certainty_threshold(model, X, percentile):
  """
  returns the certainty threshold.
  """
  certainties = get_certainties(model, X)
  threshold = np.percentile(certainties, percentile)
  return threshold

def get_certainties(model, X):
  return map(lambda p: abs(p[1]-0.5)*2, model.predict_proba(X))

def get_certainty(model, x):
  """
  returns the certainty on the prediction of the instance |x| given |model|
  """
  probabilities = model.predict_proba(x.reshape(1,-1))[0]
  certainty = abs(probabilities[1]-0.5)*2
  return certainty

def discretize_certainties(certainties, threshold):
  assert(0 < threshold < 1)
  return np.array(map(lambda x: int(x >= threshold), certainties))

def filter_uncertain(indices, y_certainty):
  return filter(lambda x: y_certainty[x] == UNCERTAIN, indices)

def filter_with_explanations(indices, X, explanations):
  for feature_id, bound in explanations:
    lower, upper = bound
    indices = filter(lambda i: within_range(X[i][feature_id], lower, upper), indices)
  return indices

def get_values_of_indices(indices, V):
  return map(lambda i: V[i], indices)


def explain_certainty(model, X_train, y_train, feature_names, instance,
                      num_features=NUM_EXPLATIONS, silent=False, html=False):
  """ |explain_certainty| is a legacy function left for compatibility.
  """
  class_names = ["uncertain", "certain"]
  explainer = LimeTabularExplainer(X_train, training_labels=y_train,
                                   feature_names=feature_names,
                                   class_names=class_names,
                                   discretize_continuous=True,
                                   discretizer=DISCRETIZER)

  explanation = explainer.explain_instance(instance, model.predict_proba,
                                           num_features=num_features,
                                           top_labels=None)
  if html:
    exp_html = explanation.as_html()
    write("uncertainty-exp.html", exp_html)

  if not silent:
    print_lime_model_prediction(model.predict_proba, instance)
    print_explanation(explanation)
    print_instance_values(feature_names, explanation, instance)

  exp_map = explanation.as_map()[1]
  exp_feature_ids = map(lambda x: x[0], exp_map)
  exp_list = explanation.as_list()
  exps = map(lambda x: parse_discretized_feature(x[0]), exp_list)
  return zip(exp_feature_ids, exps)

def print_lime_model_performance(lime_model, data_manager, y_certainty_test):
  print "==== LIME: Predicting Uncertainty ===="
  y_certainty_predict =  lime_model.predict(data_manager.X_test_scaled_e)
  acc = accuracy_score(y_certainty_test, y_certainty_predict)
  mcc = matthews_corrcoef(y_certainty_test, y_certainty_predict)
  print "ACC {:.3}".format(acc)
  print "MCC {:.3}".format(mcc)


def print_lime_model_prediction(prediction_fn, instance):
  probabilities = prediction_fn(instance.reshape(1, -1))[0]
  print "======== LIME Model Prediction ========"
  print "  Certain: {:.3f}".format(probabilities[CERTAIN])
  print "Uncertain: {:.3f}".format(probabilities[UNCERTAIN])

def get_uncertain_exps(explanation):
  return filter(lambda x : x[1] < 0, explanation.as_list())

def print_explanation(explanation, html=False):
  if html:
    exp_html = explanation.as_html()
    write("uncertainty-exp.html", exp_html)

  exp_list = explanation.as_list()

  print "======== Uncertainty Explanations =========="
  for exp, weight in exp_list:
    print " * " + exp, "(weight: {:.3f})".format(weight)

def convert_boolean_explanation(exp):
  """DRP specific conversion"""
  lte = re.match(r'(.*) <= (.*)\.50', exp)
  gt = re.match(r'(.*) > (.*)\.50', exp)
  bwn = re.match(r'(.*)\.50 < (.*) <= (.*)\.50', exp)

  if exp == "slowCool > 0.50":
    return "slowCool = True"
  elif exp == "slowCool <= 0.50":
    return "slowCool = False"

  if bwn and is_number(bwn.group(1)) and is_number(bwn.group(3)):
    if int(bwn.group(1)) + 1 == int(bwn.group(3)):
      return bwn.group(2) + " = " + bwn.group(3)
    else:
      groups = [bwn.group(1), bwn.group(2), bwn.group(3)]
      return " <= ".join(groups)
  elif lte:
    return lte.group(1) + " <= " + lte.group(2)
  elif gt and is_number(gt.group(2)):
    if "slowCool" in gt.group(1):
      return gt.group(1) + " >= " + str(int(gt.group(2))+1)
    else:
      return gt.group(1) + " >= " + str(int(gt.group(2))+1)
  return exp

def print_explanation_drp(exp_list, display_weight=False):
  """Print formatted explanation for the DRP dataset"""
  print "======== The model is uncertain about the reaction because ... =========="
  for exp, weight in exp_list:
    if display_weight:
      w = " (+)" if (weight >= 0 ) else " (-)"
    else:
      w = ""
    print " * " + convert_boolean_explanation(exp) + w

def print_instance_values(feature_names, explanation, instance):
  exp_map = explanation.as_map()[1]
  exp_feature_ids = map(lambda x: x[0], exp_map)
  exp_features = map(lambda x: feature_names[x], exp_feature_ids)
  max_length = max(map(lambda x: len(x), exp_features))
  max_length = max(max_length, len("feature name"))

  print "============= Instance Values =============="
  template_h = "{0:{1}^" + str(max_length) + "} |  value"
  template_r = "{0:<" + str(max_length) + "} : {1: 6.4f}"
  print template_h.format("feature name", " ")
  print "-" * (max_length+22)
  for feature_id in exp_feature_ids:
    feature_value = instance[feature_id]
    print template_r.format(feature_names[feature_id], feature_value)

class TestUtil(unittest.TestCase):
  def test_filter_uncertain_instances(self):
    indices = [9, 1, 3, 5, 7]
    certainties = [0.9, 0.1, 0.3, 0.5, 0.7]
    instances = [[9], [1], [3], [5], [7]]
    y_certainty = [CERTAIN] * 10
    y_certainty[1] = y_certainty[3] = y_certainty[5] = UNCERTAIN
    u_indices = filter_uncertain(indices, y_certainty)
    self.assertEqual([1, 3, 5], u_indices)

  def test_convert_boolean_explanation(self):
    btw1 = "0.50 < numberOrg <= 1.50"
    btw2 = "1.50 < numberOrg <= 2.50"
    gt1 = "slowCool > 0.50"
    gt2 = "numberInorg > 2.50"
    lte = "numberInorg <= 1.50"

    btw1_c = convert_boolean_explanation(btw1)
    btw2_c = convert_boolean_explanation(btw2)
    gt1_c = convert_boolean_explanation(gt1)
    gt2_c = convert_boolean_explanation(gt2)
    lte_c = convert_boolean_explanation(lte)

    self.assertEqual("numberOrg = 1", btw1_c)
    self.assertEqual("numberOrg = 2", btw2_c)
    self.assertEqual("slowCool = True", gt1_c)
    self.assertEqual("numberInorg >= 3", gt2_c)
    self.assertEqual("numberInorg <= 1", lte_c)


if __name__ == "__main__":
  unittest.main()
