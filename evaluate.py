from sklearn.model_selection import train_test_split
from sklearn import model_selection
from modules.evaluate import cal_aspect_prf
# from modules.aspect.lr_model import TechAspectLRModel
from modules.aspect.model import MebeAspectGBModel
# from modules.aspect.svm_model import TechAspectSVMModel
# from modules.aspect.nb_model import HotelAspectNBModel
# from modules.aspect.eec_model import HotelAspectEECModel
# from modules.aspect.brf_model import HotelAspectBRFModel
# from modules.aspect.bbg_model import HotelAspectBBGModel
# from modules.aspect.rusb_model import HotelAspectRUSBModel
# from modules.aspect.dt_model import TechAspectDTModel
# from modules.aspect.dt2_model import MebeAspectDT2Model
# from modules.aspect.svm2_model import TechAspectSVM2Model
import numpy as np
from modules.preprocess import load_aspect_data_du_lich, preprocess
from sklearn.model_selection import KFold

if __name__ == '__main__':
    inputs, outputs = load_aspect_data_du_lich('data/raw_data/mebe_shopee.csv')
    inputs = preprocess(inputs)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(inputs, outputs, test_size=0.2, random_state=14)
    model = MebeAspectGBModel()
    # kf = KFold(n_splits=5, random_state=14, shuffle=True)
    # for train_index, test_index in kf.split(inputs, outputs):
    #     model.train(np.array(inputs)[train_index], np.array(outputs)[train_index])
    #     predicts = model.predict(np.array(inputs)[test_index])
    #     print('\t\tship\t\t\t\tgiá\t\t\t\t\tchính hãng\t\t\t\tchất lượng\t\t\t\tdịch vụ\t\t\tan toàn\t\t\tothers')
    #     X = cal_aspect_prf(np.array(outputs)[test_index], predicts, num_of_aspect=7, verbal=True)
    model.train(X_train, y_train)

    predicts = model.predict(X_test)
    # print('\tstaff, service\t\troom standard\t\tfood\t\t\t\tlocation, price\t\tfacilities')
    cal_aspect_prf(y_test, predicts, num_of_aspect=7, verbal=True)

