"""
A simple logger to save labeled instances for each round.
"""

class SimpleLogger(object):
  def __init__(self, log_filename="log.txt"):
    self.f = open(log_filename,'w',0)
    self.f.write("round, labeled_instances\n")

  def log(self, round, indices):
    indices = str(list(indices))
    row = str(round) + ", " + indices + "\n"
    self.f.write(row)

  def close_log(self):
    self.f.close()


if __name__ == "__main__":
  logger = SimpleLogger("test.txt")
  logger.log(0, [1,2,3,4])
  logger.log(1, set([5,6,7]))
  logger.log(2, set([7,8,9]))
