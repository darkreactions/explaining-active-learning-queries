import sklearn, copy, argparse
from random import randint, choice
from sklearn import svm

from certainty import *
from util import *
from batch_selection import select_batch

from libact.base.dataset import Dataset
from libact.models import SVM

DATA_TRAIN = "data/toy_eal_train.csv"
DATA_TEST = "data/toy_eal_test.csv"
DATA_ALL = "data/toy_eal_all.csv"      # train + pool

ROUNDS = 10
CANDIDATES = 4
KERNEL = "poly"

AL, EAL, PL = range(3)
STRATEGIES = range(3)
STRATEGIY_NAMES = ["AL", "EAL", "PL"]

def make_active_learning_dataset(n, X, y):
  X_train = copy.deepcopy(X)
  y_train = [None] * len(y)
  for i in range(n):
    y_train[i] = y[i]
  return Dataset(X_train, y_train)

def compute_acc_mcc(model, X_test, y_test):
  acc = sklearn.metrics.accuracy_score(y_test, model.predict(X_test))
  mcc = sklearn.metrics.matthews_corrcoef(y_test, model.predict(X_test))
  return acc, mcc

def quadrant(x, y):
  if x >= 0 and y >= 0:
    return "Q1"
  elif x <= 0 and y >= 0:
    return "Q2"
  elif x <= 0 and y <= 0:
    return "Q3"
  else:
    return "Q4"

def main():
  X_train, y_train = load_data(DATA_TRAIN)
  X_test, y_test = load_data(DATA_TEST)
  X_all, y_all = load_data(DATA_ALL)

  trn_ds_eal = make_active_learning_dataset(len(y_train), X_all, y_all)
  trn_ds_al = copy.deepcopy(trn_ds_eal)
  trn_ds_pl = copy.deepcopy(trn_ds_eal)
  svm_model = SVM(kernel=KERNEL, probability=True)

  trn_datasets = [trn_ds_al, trn_ds_eal, trn_ds_pl]
  accs_list = [[], [], []]
  mccs_list = [[], [], []]

  for strategy in STRATEGIES:
    trn_ds = trn_datasets[strategy]
    svm_model.train(trn_ds)
    acc, mcc = compute_acc_mcc(svm_model.model, X_test, y_test)
    accs_list[strategy].append(acc)
    mccs_list[strategy].append(mcc)

  for i in range(ROUNDS):
    for strategy in STRATEGIES:
      trn_ds = trn_datasets[strategy]
      svm_model.train(trn_ds)
      pool_indices, X_pool = zip(*trn_ds.get_unlabeled_entries())
      pool_indices = list(pool_indices)
      certainties = get_certainties(svm_model.model, X_pool)

      if strategy == AL:
        query_indices = select_batch(1, pool_indices, X_pool, certainties,
                                     "q-best")
        query_index = query_indices[0]
        x1, x2 = X_all[query_index]

      elif strategy == EAL:
        query_indices = select_batch(CANDIDATES, pool_indices, X_pool,
                                     certainties, "k-means-uncertain")
        query_indices_q2_q4 = []
        for q in query_indices:
          x1, x2 = X_all[q]
          if quadrant(x1, x2) in ["Q2", "Q4"]:
            query_indices_q2_q4.append(q)

        if query_indices_q2_q4:
          query_indices = query_indices_q2_q4

        query_index = query_indices[randint(0, len(query_indices)-1)]

      elif strategy == PL:
        query_index = choice(pool_indices)
        x1, x2 = X_all[query_index]

      trn_ds.update(query_index, y_all[query_index])
      svm_model.train(trn_ds)
      acc, mcc = compute_acc_mcc(svm_model.model, X_test, y_test)
      accs_list[strategy].append(acc)
      mccs_list[strategy].append(mcc)

  for strategy in STRATEGIES:
    strategy_name = STRATEGIY_NAMES[strategy]
    accs_list[strategy] = map(lambda x: pretty_float(x), accs_list[strategy])
    mccs_list[strategy] = map(lambda x: pretty_float(x), mccs_list[strategy])
    print "{0}_ACC,".format(strategy_name) + ",".join(accs_list[strategy])
    print "{0}_MCC,".format(strategy_name) + ",".join(mccs_list[strategy])


if __name__ == "__main__":
  main()
