import os
import sys
import getopt
from . import utils_qna_kb

FAQ_TRAINING_DATA = os.environ.get("FAQ_TRAINING_DATA")
USE_MODEL = os.environ.get("USE_MODEL")
FAQ_INDEX_PATH = os.environ.get("FAQ_INDEX_PATH")
FAQ_TRAIN_BATCH_SIZE = int(os.environ.get("FAQ_TRAIN_BATCH_SIZE"))

argv = sys.argv[1:]

if __name__ == "__main__":    
    opts, args = getopt.getopt(argv, 'q:')
    # Check if the options' length is 2 (can be enhanced)
    if len(opts) == 0 and len(opts) > 1:
      print ('usage: predict_qna_kb.py -q <query>')  
    question = opts[0][1]
    utils = utils_qna_kb.UtilsQnAFAQ()
    ans = utils.askfaq(question,num_results=1)
    print(ans)
