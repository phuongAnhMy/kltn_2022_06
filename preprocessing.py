import sys

import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from Input_Output import Input, Output
from feature_extraction import Chi2
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator

punctuations = list(string.punctuation)
useless_labels = ['295', '296', '314', '315', '329', '330', '348', '349']


def is_nan(s):
    return s != s


def contains_punctuation(s):
    for c in s:
        if c in punctuations:
            return True
    return False


def contains_digit(w):
    for i in w:
        if i.isdigit():
            return True
    return False


def typo_trash_labeled(lst):
    for i in lst:
        if i in useless_labels:
            return True
    return False


def load_data(data_path, num_aspects):
    if num_aspects == 6:
        categories = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn']
    else:
        categories = ['cấu hình', 'mẫu mã', 'hiệu năng', 'ship', 'giá', 'chính hãng', 'dịch vụ', 'phụ kiện']

    inputs, outputs = [], []
    df = pd.read_csv(data_path, encoding='utf-8')
    aspects = list(range(num_aspects))
    # print(df.iterrows())
    for index, row in df.iterrows():
        if is_nan(row['text']) == 0:
            text = row['text'].strip()
            inputs.append(Input(text))
            # print(row['label'])
            _scores = list(row['label'][1:-1].split(', '))
            scores = [int(i) for i in _scores[:num_aspects]]
            outputs.append(Output(aspects, scores))
    print(np.array(outputs).shape)
    return inputs, outputs


def make_vocab(inputs):
    # """
    cv = CountVectorizer()
    x = cv.fit_transform(inputs)
    vocab = cv.get_feature_names()
    with open(r"/Users/minhdam/Desktop/tech_shopee 2/data/chi2/data_{}/{}_vocab.txt".format(str(sys.argv[1])[0:4],
                                                                                            str(sys.argv[1])), 'w',
              encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))
    # """

    vocab = []
    for ip in inputs:
        text = ip.text.split(' ')
        for token in text:
            vocab.append(token)
    # Make a non-duplicated vocabulary
    vocab = list(dict.fromkeys(vocab))

    with open(r"/Users/minhdam/Desktop/tech_shopee 2/data/data_{}/{}_vocab.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), 'w', encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))

    return vocab


def preprocess_inputs(inputs, outputs, text_len, num_aspects):
    inp, outp = [], []
    for ip, op in zip(inputs, outputs):
        text = ip.text.strip().split(' ')
        if len(text) <= text_len:
            for j in range(len(text)):
                if contains_digit(text[j].strip()):
                    text[j] = '0'
            for token in text:
                if len(token) <= 1 or token.strip() in punctuations:
                    text.remove(token)
            ip.text = ' '.join(text)
            inp.append(ip.text)
            outp.append(op.scores)

    # le = []
    # for ip in inp:
    #     le.append(len(ip.split(' ')))
    # x = Counter(le).keys()
    # y = Counter(le).values()
    # print(max(x))
    # plt.bar(x, y)
    # plt.show()

    # for i in range(6):
    #     li = []
    #     for ip, op in zip(inp, outp):
    #         if op[i] == 1:
    #             li.append(ip)
    #     text = " ".join(i for i in li)
    #     wcl = WordCloud(background_color='white').generate(text)
    #     plt.imshow(wcl)
    #     plt.show()

    vocab = make_vocab(inp)
    print('Oh')
    df = Chi2(inp, outp, num_aspects)
    return inp, outp, vocab, df


def load_chi2(path):
    dictionary = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            t = line.strip().split(' ')
            dictionary[t[0]] = float(t[2])

    return dictionary


datasets = {'mebeshopee': [6, 0],
            'mebetiki': [6, 1],
            'techshopee': [8, 2],
            'techtiki': [8, 3]
            }
data_paths = [
    r"/Users/minhdam/Downloads/review/data/mebe_shopee.csv",
    r"/Users/minhdam/Downloads/review/data/mebe_tiki.csv",
    r"/Users/minhdam/Downloads/review/data/tech_shopee.csv",
    r"/Users/minhdam/Downloads/review/data/tech_tiki.csv.csv",
]
text_len = [100, 100, 100, 100]  # 30, 38, 55, 52

if __name__ == '__main__':
    argv = sys.argv[1]
    data_path = data_paths[datasets[argv][1]]
    num_aspects = datasets[argv][0]

    # Load inputs, outputs from data file
    inputs, outputs = load_data(data_path, num_aspects)
    inputs, outputs, vocab = preprocess_inputs(inputs, outputs, text_len[datasets[argv][1]], num_aspects)
