import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import string
from Input_Output import Input, Output
from sklearn.feature_extraction.text import CountVectorizer
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
            scores = [0 if row['aspect{}'.format(i)] == 0 else 1 for i in range(1, 9)]
            outputs.append(Output(aspects, scores))

    return inputs, outputs


# def Chi2(inputs, outputs, num_aspects):
#     if num_aspects == 6:
#         categories = ['Ship', 'Gia', 'Chinh hang', 'Chat luong', 'Dich vu', 'An toan']
#     else:
#         categories = ['Cau hinh', 'Mau ma', 'Hieu nang', 'Ship', 'Gia', 'Chinh hang', 'Dich vu', 'Phu kien']
#     cv = CountVectorizer()
#     x = cv.fit_transform(inputs)
#
#     y = []
#     skb = [SelectKBest(chi2, k='all') for _ in range(num_aspects)]
#     # skb = SelectKBest(chi2, k='all')
#     res = []
#     for i in range(num_aspects):
#         y = [op[i] for op in outputs]
#         _chi2 = skb[i].fit_transform(x, y)
#
#         feature_names = cv.get_feature_names()
#         _chi2_scores = skb[i].scores_
#         _chi2_pvalues = skb[i].pvalues_
#
#         chi2_dict = {'word': feature_names, 'score': list(_chi2_scores), 'pvalue': list(_chi2_pvalues)}
#         df = pd.DataFrame(chi2_dict, columns=['word', 'score', 'pvalue'])
#         df = df.sort_values('score', ascending=False)
#         res.append(df)
#         with open("data/chi2/tech_tiki/chi2/tech_tiki_vocab_{}.txt".format(i), 'w', encoding='utf8') as f:
#             for w, s, p in zip(df['word'], df['score'], df['pvalue']):
#                 f.write('{} \t {} \t {}\n'.format(w, s, p))
#         # df.to_csv("data/chi2/tech_tiki/chi2/tech_tiki_vocab_{}.csv".format(i), encoding='utf-8')
#     return res


def preprocess_inputs(inputs, outputs, text_len, num_aspects):
    inp, outp = [], []
    for ip, op in zip(inputs, outputs):
        # print(type(ip), ' ', ip.strip())
        text = str(ip).strip().split(' ')
        # text = ip.split(' ')
        if len(text) <= text_len:
            for j in range(len(text)):
                if contains_digit(text[j].strip()):
                    text[j] = '0'
            for token in text:
                if len(token) <= 1 or token.strip() in punctuations:
                    text.remove(token)
            ip = ' '.join(text)
            inp.append(ip)
            outp.append(op.scores)
    # vocab = make_vocab(inp)
    # df = Chi2(inp, outp, num_aspects)
    return inp, outp

# inputs, outputs = load_data('data/raw_data/tech_tiki.csv', 8)
# inputs, outputs, ndf = preprocess_inputs(inputs, outputs, 50, 8)