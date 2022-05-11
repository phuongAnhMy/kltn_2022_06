import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
import pickle
import lightgbm
import lightgbm as lgb
from calchi2 import preprocess_inputs
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# from datagen import GenerateData
from models import AspectOutput, Input
from modules.models import Model
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


class MebeAspectRFModel(Model):
    def __init__(self):
        self.NUM_OF_ASPECTS = 9
        self.vocabs = []
        self.vocabs_lda_1 = []
        self.vocabs_lda_2 = []
        self.vocabs_lda_3 = []
        self.vocabs_lda_4 = []
        self.vocabs_lda_5 = []
        self.vocabs_lda_6 = []
        self.vocabs_lda_7 = []
        self.vocabs_lda_8 = []
        for i in range(self.NUM_OF_ASPECTS - 1):
            vocab = []
            # with open('data/vocab/tech_shopee_vocab_{}.txt'.format(i), "rb") as new_filename:
            #     vocab = pickle.load(new_filename)
            # with open('data/chi2/tech_tiki/chi2/tech_tiki_vocab_{}.txt'.format(i),
            #           encoding='utf-8') as new_filename:
            #     for line in new_filename:
            #         tmp = line.split(' ')
            #         vocab.append([tmp[0], float(tmp[2])])
            # vocab_new = []
            # for w in vocab:
            #     if w[1] > 10:
            #         vocab_new.append(w)
            # self.vocabs.append(vocab_new)
            with open('data/chi2/tech_tiki/chi2/tech_tiki_vocab_{}.txt'.format(i), 'rb') as new_filename:
                vocab = pickle.load(new_filename)
                vocab_new = []
                for w in vocab:
                    if float(w[1]) > 10:
                        vocab_new.append(w)
                self.vocabs.append(vocab_new)
            print(self.vocabs)

            with open('data/chi2/tech_tiki/chi2/chi2_tech_loose/tech_tiki_vocab_{}_test_1.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_1 = []
                for w in vocab_lda:
                    if float(w[1]) > 5:
                        vocab_new_lda_1.append(w)

                self.vocabs_lda_1.append(vocab_new_lda_1)
            with open('data/chi2/tech_tiki/chi2/chi2_tech_loose/tech_tiki_vocab_{}_test_2.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_2 = []
                for w in vocab_lda:
                    if float(w[1]) > 5:
                        vocab_new_lda_2.append(w)

                self.vocabs_lda_2.append(vocab_new_lda_2)
            with open('data/chi2/tech_tiki/chi2/chi2_tech_loose/tech_tiki_vocab_{}_test_3.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_3 = []
                for w in vocab_lda:
                    if float(w[1]) > 10:
                        vocab_new_lda_3.append(w)

                self.vocabs_lda_3.append(vocab_new_lda_3)
            with open('data/chi2/tech_tiki/chi2/chi2_tech_loose/tech_tiki_vocab_{}_test_4.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_4 = []
                for w in vocab_lda:
                    if float(w[1]) > 10:
                        vocab_new_lda_4.append(w)

                self.vocabs_lda_4.append(vocab_new_lda_4)
            with open('data/chi2/tech_tiki/chi2/chi2_tech_loose/tech_tiki_vocab_{}_test_5.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_5 = []
                for w in vocab_lda:
                    if float(w[1]) > 10:
                        vocab_new_lda_5.append(w)

                self.vocabs_lda_5.append(vocab_new_lda_5)
        print(self.vocabs_lda_5)
        # self.models = [lgb.LGBMClassifier(random_state=14) for _ in range(self.NUM_OF_ASPECTS - 1)]
        # self.models = [lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', learning_rate= 0.09,
        #                                   metric = 'binary_logloss', max_depth = 10,random_state=14) for _ in
        #                range(self.NUM_OF_ASPECTS - 1)]
        # self.vocabs = np.array(self.vocabs)
        # print(type(self.vocabs[0]))
        self.models = [GradientBoostingClassifier(random_state=14) for _ in range(self.NUM_OF_ASPECTS - 1)]
        # self.models = [XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=3,subsample=0.8,colsample_bytree=0.5,gamma=1,random_state=14) for _ in range(self.NUM_OF_ASPECTS - 1)]


    def _represent(self, inputs, i):
        """

        :param list of models.Input inputs:
        :return:
        """

        features = []
        for ip in inputs:
            _feature = [1 if str(vocab[0]) in str(ip).split(' ') else 0 for vocab in self.vocabs[i]]
            # _feature = [1 if str(vocab[0]) in ip else 0 for vocab in self.vocabs[i]]
            features.append(_feature)
        # features.append(feature)

        return features

    def _represents(self, inputs):
        """

        :param list of models.Input inputs:
        :return:
        """
        features = []
        # print(self.vocabs[0][0][0])
        for i in range(self.NUM_OF_ASPECTS - 1):
            _features = []

            for ip in inputs:
                _feature = [1 if str(vocab[0]) in str(ip).split(' ') else 0 for vocab in self.vocabs[i]]
                _feature_lda_1 = [3 if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_1[i]]
                _feature_lda_2 = [vocab[1] if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_2[i]]
                _feature_lda_3 = [vocab[1] if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_3[i]]
                _feature_lda_4 = [vocab[1] if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_4[i]]
                _feature_lda_5 = [vocab[1] if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_5[i]]
                _feature_list = [
                                # _feature_lda_1,
                                 _feature_lda_2,
                                 # _feature_lda_3,
                                 # _feature_lda_4,
                                 # _feature_lda_5
                                 ]
                for f in _feature_list:
                    _feature += f
                _features.append(_feature)
                # _feature = [1 if v in ip.text else 0 for v in self.vocabs[i]]
            features.append(_features)
        return features

    def chi2vocabs(self, ndf):
        for i in range(self.NUM_OF_ASPECTS - 1):
            vocab = []
            for k in range(len(ndf[i])):
                if (ndf[i].iat[k, 1]) > 0:
                    vocab.append([ndf[i].iat[k, 0], ndf[i].iat[k, 1]])
            self.vocabs.append(vocab)
        # print(len(self.vocabs[0]))

    def load_data(self, input, output):
        # if self.NUM_OF_ASPECTS == 6:
        #     categories = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn']
        # else:
        #     categories = ['cấu hình', 'mẫu mã', 'hiệu năng', 'ship', 'giá', 'chính hãng', 'dịch vụ', 'phụ kiện']
        inputs, outputs = [], []
        df = pd.DataFrame({
            'text': input,
            'label': output
        })
        df = df.astype({'label': str})
        # print(df)
        aspects = list(range(self.NUM_OF_ASPECTS - 1))
        # print(df.iterrows())
        for index, row in df.iterrows():
            text = row['text'].strip()
            inputs.append(Input(text))
            # print(row['label'])
            _scores = list(row['label'][1:-1].split(', '))
            print(_scores)
            scores = [int(i) for i in _scores[:self.NUM_OF_ASPECTS - 1]]
            # print(scores)
            outputs.append(AspectOutput(aspects, scores))
        # print(np.array(outputs).shape)
        return inputs, outputs

    def train(self, inputs, outputs):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        # X = self._represent(inputs)
        # print(len(X[0][0]))
        # print(type(outputs[0]))
        # ys = [np.array([output.scores[i] for i in range(self.NUM_OF_ASPECTS - 1)] for output in outputs)]
        # print(ys)

        _input , _output = self.load_data(inputs, outputs)
        _inputs, _outputs = preprocess_inputs(inputs, _output, 100, self.NUM_OF_ASPECTS - 1)
        # self.chi2vocabs(ndf)
        # print(_output[0])
        # for i in range(self.NUM_OF_ASPECTS - 1):
        #     print(self.vocabs[i])
        # print(_input)
        X = self._represents(inputs)


        ys = [np.array([output.get_score()[i] for output in outputs]) for i in range(self.NUM_OF_ASPECTS - 1)]
        # for i in range(self.NUM_OF_ASPECTS - 1):
        #     for output in outputs:
        #         l = output.get_score()
        #         print(l[0])
        #         break
        oversample = RandomOverSampler(sampling_strategy=0.7, random_state=20)
        # for i in range(self.NUM_OF_ASPECTS - 1):
        #     count1 = sum(ys[i])
        #     count0 = len(ys[i]) - count1
        #     print(i, count0, count1)
        #     if count0 * 0.7 > count1 != 0:
        #         Xn, ysn = oversample.fit_resample(X[i], ys[i])
        #         self.models[i].fit(Xn, ysn)
        #     else:
        #         self.models[i].fit(X[i], ys[i])

        for i in range(self.NUM_OF_ASPECTS - 1):
            self.models[i].fit(X[i], ys[i])
            print(np.array(X[i]).shape)
        #New
        # for i in range(self.NUM_OF_ASPECTS - 1):
        #     # X, ysn = handle.generate(inputs.copy(), ys[i])
        #     # ysn = ys[i].copy()
        #     count0 = 0
        #     count1 = 0
        #     for j in range(len(ys[i])):
        #         if ys[i][j] == 0:
        #             count0 += 1
        #         else:
        #             count1 += 1
        #     # print(count0 / count1)
        #     # self.models[i].fit(X[i], ys[i])
        #     if max(count0, count1) / min(count0, count1) > 2.2:
        #         # generator
        #         # print(count1, '(0)', i)
        #         # file = open('data/1604_new_rv.txt', 'a')
        #         # file.write('ASPECT {}'.format(i))
        #         # file.write('\n')
        #         # file.close()
        #         # Xt, ysn, k = GenerateData().generate(inputs.copy(), ys[i])
        #         # print(i, ' ', sum(k) / len(k), ' ', len(k), ' ', max(k), ' ', min(k))
        #         # Xn = self._represent(Xt, i)
        #         # self.models[i].fit(Xn, ysn)
        #
        #         # duplicate
        #         print(max(count0, count1) / min(count0, count1))
        #         Xt = self._represent(inputs.copy(), i)
        #         Xn, ysn = oversample.fit_resample(Xt, ys[i])
        #         self.models[i].fit(Xn, ysn)
        #     # self.models[i].fit(X[i], ys[i])
        #     else:
        #         Xn = self._represent(inputs.copy(), i)
        #         self.models[i].fit(Xn, ys[i])

        # for i in range(self.NUM_OF_ASPECTS - 1):
        #     count0 = 0
        #     count1 = 0
        #     for j in range(len(ys[i])):
        #         if ys[i][j] == 0:
        #             count0 += 1
        #         else:
        #             count1 += 1
        #     # print(count0, count1)
        #     # self.models[i].fit(X[i], ys[i])
        #     if max(count0, count1) / min(count0, count1) > 1.8:
        #         Xn, ysn = oversample.fit_resample(X[i], ys[i])
        #         self.models[i].fit(Xn, ysn)
        #     # self.models[i].fit(X[i], ys[i])
        #     else:
        #         self.models[i].fit(X[i], ys[i])

    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.AspectOutput
        """
        # outputs = []
        # _inputs, _outputs, ndf = preprocess_inputs(inputs, outputs, 100, self.NUM_OF_ASPECTS - 1)
        # X = self._represent(inputs)
        X = []
        X = self._represents(inputs.copy())
        # for i in range(self.NUM_OF_ASPECTS - 1):
        #     Xn = self._represents(inputs.copy())
        #     print(Xn)
        #     X.append(Xn)
        #     break
        outputs = []
        predicts = [self.models[i].predict(X[i]) for i in range(self.NUM_OF_ASPECTS - 1)]
        # print(predicts)
        for ps in zip(*predicts):
            labels = list(range(self.NUM_OF_ASPECTS))
            scores = list(ps)
            if 1 in scores:
                scores.append(0)
            else:
                scores.append(1)
            outputs.append(AspectOutput(labels, scores))

        return outputs
