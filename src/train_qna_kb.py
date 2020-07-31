import os
from utils_qna_kb import train, ask


if __name__ == "__main__":    
    sentence_dict = train(data_csv_path = '/home/sandipan/projects/conversation-ai/data/faq.csv')
    # print(sentence_dict)