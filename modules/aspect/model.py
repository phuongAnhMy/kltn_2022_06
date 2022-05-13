import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
import pickle
from calchi2 import preprocess_inputs
from models import AspectOutput, Input
from modules.models import Model
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


class MebeAspectGBModel(Model):
    def __init__(self):
        self.NUM_OF_ASPECTS = 7
        self.vocabs = []
        self.vocabs_lda_1 = []
        self.vocabs_lda_2 = []
        self.vocabs_lda_3 = []
        self.vocabs_lda_4 = []
        self.vocabs_lda_5 = []

        for i in range(self.NUM_OF_ASPECTS - 1):
            vocab = []
            # with open('data/vocab/tech_shopee_vocab_{}.txt'.format(i), "rb") as new_filename:
            #     vocab = pickle.load(new_filename)
            # with open('data/chi2/tech/chi2/tech_tiki_vocab_{}.txt'.format(i),
            #           encoding='utf-8') as new_filename:
            #     for line in new_filename:
            #         tmp = line.split(' ')
            #         vocab.append([tmp[0], float(tmp[2])])
            # vocab_new = []
            # for w in vocab:
            #     if w[1] > 10:
            #         vocab_new.append(w)
            # self.vocabs.append(vocab_new)
            # with open('data/chi2/mebe_shopee/chi2/mebe_shopee_vocab_{}.txt'.format(i), 'rb') as new_filename:
            with open('data/chi2/mebe_shopee/chi2/mebe_shopee_vocab_{}.txt'.format(i),
                      encoding='utf-8') as new_filename:
                for line in new_filename:
                    tmp = line.split(' ')
                    vocab.append([tmp[0], float(tmp[2])])

                vocab_new = []
                for w in vocab:
                    if float(w[1]) > 10:
                        vocab_new.append(w)
                self.vocabs.append(vocab_new)
            print(self.vocabs)

            with open('data/chi2/mebe_shopee/chi2/chi2_selected/mebe_shopee_vocab_{}_test_1.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_1 = []
                for w in vocab_lda:
                    if float(w[1]) > 10:
                        vocab_new_lda_1.append(w)

                self.vocabs_lda_1.append(vocab_new_lda_1)
            with open('data/chi2/mebe_shopee/chi2/chi2_selected/mebe_shopee_vocab_{}_test_2.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_2 = []
                for w in vocab_lda:
                    if float(w[1]) > 10:
                        vocab_new_lda_2.append(w)

                self.vocabs_lda_2.append(vocab_new_lda_2)
            with open('data/chi2/mebe_shopee/chi2/chi2_selected/mebe_shopee_vocab_{}_test_3.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_3 = []
                for w in vocab_lda:
                    if float(w[1]) > 10:
                        vocab_new_lda_3.append(w)

                self.vocabs_lda_3.append(vocab_new_lda_3)
            with open('data/chi2/mebe_shopee/chi2/chi2_selected/mebe_shopee_vocab_{}_test_4.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_4 = []
                for w in vocab_lda:
                    if float(w[1]) > 10:
                        vocab_new_lda_4.append(w)

                self.vocabs_lda_4.append(vocab_new_lda_4)
            with open('data/chi2/mebe_shopee/chi2/chi2_selected/mebe_shopee_vocab_{}_test_5.txt'.format(i), "rb") as new_filename:
                vocab_lda = pickle.load(new_filename)
                vocab_new_lda_5 = []
                for w in vocab_lda:
                    if float(w[1]) > 10:
                        vocab_new_lda_5.append(w)

                self.vocabs_lda_5.append(vocab_new_lda_5)
        print(self.vocabs_lda_5)
        self.models = [GradientBoostingClassifier(random_state=14) for _ in range(self.NUM_OF_ASPECTS - 1)]
        # self.models = [XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=3,subsample=0.8,colsample_bytree=0.5,gamma=1,random_state=14) for _ in range(self.NUM_OF_ASPECTS - 1)]


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
                _feature_lda_1 = [2 if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_1[i]]
                _feature_lda_2 = [2 if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_2[i]]
                _feature_lda_3 = [2 if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_3[i]]
                _feature_lda_4 = [2 if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_4[i]]
                _feature_lda_5 = [2 if str(vocab[0]) in ip else 0 for vocab in self.vocabs_lda_5[i]]
                _feature_list = [
                                # _feature_lda_1,
                                #  _feature_lda_2,
                                #  _feature_lda_3,
                                #  _feature_lda_4,
                                 _feature_lda_5
                                 ]
                for f in _feature_list:
                    _feature += f
                _features.append(_feature)
                # _feature = [1 if v in ip.text else 0 for v in self.vocabs[i]]
            features.append(_features)
        return features

    # def chi2vocabs(self, ndf):
    #     for i in range(self.NUM_OF_ASPECTS - 1):
    #         vocab = []
    #         for k in range(len(ndf[i])):
    #             if (ndf[i].iat[k, 1]) > 0:
    #                 vocab.append([ndf[i].iat[k, 0], ndf[i].iat[k, 1]])
    #         self.vocabs.append(vocab)


    def load_data(self, input, output):
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

        _input , _output = self.load_data(inputs, outputs)
        _inputs, _outputs = preprocess_inputs(inputs, _output, 100, self.NUM_OF_ASPECTS - 1)

        X = self._represents(inputs)


        ys = [np.array([output.get_score()[i] for output in outputs]) for i in range(self.NUM_OF_ASPECTS - 1)]


        for i in range(self.NUM_OF_ASPECTS - 1):
            self.models[i].fit(X[i], ys[i])
            print(np.array(X[i]).shape)

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
