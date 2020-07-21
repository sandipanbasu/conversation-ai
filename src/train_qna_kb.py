import os
from . import utils_qna_kb

FAQ_TRAINING_DATA = os.environ.get("FAQ_TRAINING_DATA")
USE_MODEL = os.environ.get("USE_MODEL")
FAQ_INDEX_PATH = os.environ.get("FAQ_INDEX_PATH")
FAQ_TRAIN_BATCH_SIZE = int(os.environ.get("FAQ_TRAIN_BATCH_SIZE"))
# from utils_qna_kb import load_data, load_USE_model, build_search_index

if __name__ == "__main__":
    sentence_dict = utils_qna_kb.train_faq_kb(data_csv_path = FAQ_TRAINING_DATA,
                                            USE_path = USE_MODEL,
                                            faq_train_batch_size = FAQ_TRAIN_BATCH_SIZE)
    # print(sentence_dict)