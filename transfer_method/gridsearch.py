from sklearn.model_selection import GridSearchCV, KFold
from model import base_feature_1
from keras.wrappers.scikit_learn import KerasClassifier
from feature import com_seq_feature
from dataloader import load_train_test
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# data
tgt = load_train_test(max_size=99, split=False)
t_ = load_train_test(isencoder=False, split=False)
tgt_f = com_seq_feature(t_[0])

X = np.concatenate([tgt[0], tgt_f], axis=1)
y = np.array(tgt[1])

model = KerasClassifier(build_fn=base_feature_1)

lr = [0.01, 0.001, 3e-4]
ed = [50, 100, 150]
ps = [3, 5]
fd_2 = [32, 64]

param_grid = dict(lr=lr, ed=ed, ps=ps, fd_2=fd_2)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(shuffle=True), verbose=10, scoring='roc_auc')
grid_result = grid.fit(X, y, batch_size=64, epochs=150)

print('Best : {}, using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{},{} with : {}'.format(mean, stdev, param))
