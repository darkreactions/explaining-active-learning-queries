import copy, random, time, argparse
import numpy as np

from certainty import *
from batch_selection import *
from util import *
from puk_kernel import PUK_kernel
from interpretable_data_manager import *
from logger import SimpleLogger

import sklearn
from sklearn import svm
from sklearn.metrics import matthews_corrcoef, accuracy_score

from lime.lime_tabular import LimeTabularExplainer

# libact classes for active learning
from libact.base.dataset import Dataset
from libact.models import SVM
from libact.query_strategies import UncertaintySampling, RandomSampling

TRAINING_DATA_PATH = "data/Supp_Info_Train_formatted.csv"
TEST_DATA_PATH = "data/Supp_Info_Test_formatted.csv"
NO_ELEM_TRAINING = "data/Supp_Info_Train_formatted_no_elements.csv"
NO_ELEM_TEST = "data/Supp_Info_Test_formatted_no_elements.csv"

KERNEL = PUK_kernel
EAL, AL, PL = STRATEGIES = range(3)
SHOW_LIME = False
INTERACTIVE = True
MAX_EXP_FEATURE = 22

# default argument values
AL_ROUNDS = 3
INITIAL_INSTANCES = 100
THRESHOLD = 30
NUM_FEATURES = 5
BATCH_SIZE = 20
LOG_FILE = "drp_eal_log.txt"

def make_active_learning_dataset(X, y, n_labeled=5, n_classes=2):
  n_samples = len(y)
  X_train = copy.deepcopy(X)
  y_train = [None] * n_samples

  # sample until labeled instances have both classes
  label_indices = random.sample(range(n_samples), n_labeled)
  label_classes = set(map(lambda i: y[i], label_indices))
  while len(label_classes) < n_classes:
    label_indices = random.sample(range(n_samples), n_labeled)
    label_classes = set(map(lambda i: y[i], label_indices))

  for i in label_indices:
    y_train[i] = y[i]

  trn_ds = Dataset(X_train, y_train)
  labeled_trn_ds = Dataset(X, y)
  return trn_ds, labeled_trn_ds

class DataManager(object):
  def __init__(self):
    self.feature_names = get_feature_names(TRAINING_DATA_PATH)
    self.feature_names_e = get_feature_names(NO_ELEM_TRAINING)
    self.class_names = ["failure", "success"]
    self.load_dataset()
    self.init_feature_id_map()

  def load_dataset(self):
    X_train, y_train = load_data(TRAINING_DATA_PATH)
    X_test, y_test = load_data(TEST_DATA_PATH)
    scaler = StandardScaler().fit(X_train)
    self.scaler = scaler
    self.X_train_scaled = scaler.transform(X_train)
    self.X_test_scaled = scaler.transform(X_test)
    self.X_train, self.y_train = X_train, y_train
    self.X_test, self.y_test = X_test, y_test
    self.trn_ds_eal, _ = make_active_learning_dataset(self.X_train_scaled,
                                                      y_train,
                                                      INITIAL_INSTANCES)
    self.trn_ds_al = copy.deepcopy(self.trn_ds_eal) # for active learning
    self.trn_ds_pl = copy.deepcopy(self.trn_ds_eal) # for passive learning
    self.trn_ds_list = [self.trn_ds_eal, self.trn_ds_al, self.trn_ds_pl]

    self.X_train_e, _ = load_data(NO_ELEM_TRAINING)
    self.X_test_e,  _ = load_data(NO_ELEM_TEST)
    self.scaler_e = StandardScaler().fit(self.X_train_e)
    self.X_train_scaled_e = self.scaler_e.transform(self.X_train_e)
    self.X_test_scaled_e = self.scaler_e.transform(self.X_test_e)

  def init_feature_id_map(self):
    feature_id_map = {}
    for i, feature in enumerate(self.feature_names_e):
      j = self.feature_names.index(feature)
      feature_id_map[i] = j

    for key, val in feature_id_map.iteritems():
      assert(self.feature_names_e[key] == self.feature_names[val])

    self.feature_id_map = feature_id_map

  def get_labeled_indices(self):
    labeled_indices = [idx for idx, entry in enumerate(self.trn_ds_eal.data)
                       if entry[1] is not None]
    return labeled_indices

def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--active_learning_rounds", type=int,
                      default=AL_ROUNDS)
  parser.add_argument("-i", "--initial_labeled_instances", type=int,
                      default=INITIAL_INSTANCES)
  parser.add_argument("-n", "--num_features", type=int,
                      default=NUM_FEATURES,
                      help="number of features to be used for LIME")
  parser.add_argument("-p", "--percentile", type=int,
                      default=THRESHOLD,
                      help="set threshold percentile")
  parser.add_argument("-b", "--batch_size", type=int,
                      default=BATCH_SIZE,
                      help="the batch size")
  parser.add_argument("-l", "--log_file", type=str,
                      default=LOG_FILE,
                      help="log file name")
  return parser.parse_args()

def register_arguments(args):
  global AL_ROUNDS, INITIAL_INSTANCES, THRESHOLD
  global LOG_FILE, NUM_FEATURES, BATCH_SIZE
  AL_ROUNDS = args.active_learning_rounds
  INITIAL_INSTANCES = args.initial_labeled_instances
  THRESHOLD = args.percentile
  NUM_FEATURES = args.num_features
  BATCH_SIZE = args.batch_size
  LOG_FILE = args.log_file

def update_accs_mccs(accs, mccs, dm, predict_fn, strategy):
  mcc = matthews_corrcoef(dm.y_test, predict_fn(dm.X_test_scaled))
  acc = accuracy_score(dm.y_test, predict_fn(dm.X_test_scaled))
  mccs[strategy].append(mcc)
  accs[strategy].append(acc)

def print_last_round_mcc(round, accs, mccs):
  print "Round {}".format(round)
  print  "EAL MCC: {:.3f}".format(mccs[EAL][-1])
  print  " AL MCC: {:.3f}".format(mccs[AL][-1])
  print  " PL MCC: {:.3f}".format(mccs[PL][-1])

def print_mcc_summary(mccs):
  pretty_float = lambda x: "{:.3f}".format(x)
  mcc_eal = map(pretty_float, mccs[EAL])
  mcc_al = map(pretty_float, mccs[AL])
  mcc_pl = map(pretty_float, mccs[PL])
  print "============= Summary ============="
  print "EAL MCC: " + ", ".join(mcc_eal)
  print " AL MCC: " + ", ".join(mcc_al)
  print " PL MCC: " + ", ".join(mcc_pl)

def run_active_learning():
  logger = SimpleLogger(LOG_FILE)
  dm = DataManager()
  im = InterpretableDataManager()
  drp_model = SVM(kernel=KERNEL, probability=True)
  lime_model = svm.SVC(kernel=KERNEL, probability=True)
  accs = [[], [], []]
  mccs = [[], [], []]

  labeled_indices = dm.get_labeled_indices()
  logger.log(0, labeled_indices)

  for strategy in STRATEGIES:
    trn_ds = dm.trn_ds_list[strategy]
    drp_model.train(trn_ds)
    update_accs_mccs(accs, mccs, dm, drp_model.model.predict, strategy)

  print_last_round_mcc(0, accs, mccs)
  assert(AL_ROUNDS <= len(dm.y_train) - INITIAL_INSTANCES)

  for round in xrange(1, AL_ROUNDS+1):
    print "================================================="
    print "Round", round
    print "================================================="
    for strategy in STRATEGIES:
      trn_ds = dm.trn_ds_list[strategy]
      exclusion = set()
      batch = set()

      unlabeled_indices, unlabeled_X_scaled = zip(*trn_ds.get_unlabeled_entries())
      certainties = get_certainties(drp_model.model, dm.X_train_scaled)
      if strategy == EAL:
        threshold = get_certainty_threshold(drp_model.model, dm.X_train_scaled, THRESHOLD)
        y_certainty = discretize_certainties(certainties, threshold)

        lime_model.fit(dm.X_train_scaled_e, y_certainty)
        if SHOW_LIME:
          certainties_test = get_certainties(drp_model.model, dm.X_test_scaled)
          y_certainty_test = discretize_certainties(certainties_test, threshold)
          print_lime_model_performance(lime_model, dm, y_certainty_test)

        while (len(batch) < BATCH_SIZE):
          query_id = query_least_confident(unlabeled_indices, certainties,
                                           exclusion)
          query = dm.X_train_scaled[query_id]
          query_unscaled = dm.X_train_e[query_id]
          instance_certainty = get_certainty(drp_model.model, query)
          print "Explaining Query with id #{:d}".format(query_id)
          print "Certainty {:.3f}".format(instance_certainty)

          explainer = LimeTabularExplainer(dm.X_train_e,
                                           training_labels=y_certainty,
                                           feature_names=dm.feature_names_e,
                                           class_names=["uncertain", "certain"],
                                           discretize_continuous=True,
                                           discretizer="entropy")

          predict_fn = lambda x: lime_model.predict_proba(dm.scaler_e.transform(x)).astype(float)

          for i in xrange(0, MAX_EXP_FEATURE, 2):
            exp = explainer.explain_instance(query_unscaled, predict_fn,
                                             num_features=NUM_FEATURES+i)
            uncertain_exp_list = get_uncertain_exps(exp)
            if (len(uncertain_exp_list) >= NUM_FEATURES - 2):
              break
            print "INFO: looping"

          if SHOW_LIME:
            print_lime_model_prediction(predict_fn, query_unscaled)

          exp_indices = get_indices_exp_region(exp, dm, unlabeled_indices,
                                               y_certainty)
          exp_instances = get_values_of_indices(exp_indices, dm.X_train_scaled)
          exp_certainties = get_values_of_indices(exp_indices, certainties)
          batch_indices = select_batch(min(BATCH_SIZE, BATCH_SIZE-len(batch)),
                                       exp_indices, exp_instances,
                                       exp_certainties, "k-means-uncertain")


          if len(batch_indices) == 0:
            exclusion.add(query_id)
            continue

          print ""
          print_explanation_drp(uncertain_exp_list, False)
          print ""
          print "Instances in the batch: {}".format(len(batch_indices))
          im.describe_instances(batch_indices)
          print ""
          im.describe_instance(query_id)
          print ""

          exclusion.update(set(exp_indices))
          if ask_expert():
            batch.update(set(batch_indices))
          else:
            print "INFO: Not including in the batch"

        logger.log(round, batch)
        print "INFO: Labeling the batch"
        label_batch(trn_ds, dm.y_train, batch)

      elif strategy == AL: # AL + k-means-uncertain
        unlabeled_X_scaled = get_values_of_indices(unlabeled_indices,
                                                   dm.X_train_scaled)
        unlabeled_certainties = get_values_of_indices(unlabeled_indices,
                                                      certainties)
        batch_indices = select_batch(BATCH_SIZE, unlabeled_indices,
                                     unlabeled_X_scaled, unlabeled_certainties,
                                     "k-means-uncertain")
        label_batch(trn_ds, dm.y_train, batch_indices)

      elif strategy == PL: # Passive Learning
        batch_indices = random.sample(unlabeled_indices, BATCH_SIZE)
        label_batch(trn_ds, dm.y_train, batch_indices)

      drp_model.train(trn_ds)
      update_accs_mccs(accs, mccs, dm, drp_model.model.predict, strategy)

  print_mcc_summary(mccs)

def ask_expert():
  if not INTERACTIVE:
    return random.random() > 0.33

  answer = None
  while not answer:
    answer = raw_input("Would you like to conduct the experiments " +
                       " for this region? (y/n) : \n")
    if answer == "y" or answer == "yes":
      return True
    elif answer == "n" or answer == "no":
      return False
    else:
      answer = None


if __name__ == '__main__':
  start_time = time.time()
  args = get_arguments()
  register_arguments(args)
  run_active_learning()
  print "Execution Time: {:.3f} seconds".format(time.time() - start_time)
