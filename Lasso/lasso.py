from mglearn.datasets import load_extended_boston

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso, Ridge

import numpy as np



boston_data, boston_target = load_extended_boston()



x_train, x_test, y_train, y_test = train_test_split(boston_data, boston_target,

                                                    random_state=0, test_size=0.3)



lasso = Lasso().fit(x_train, y_train)



print('{:.3f}'.format(lasso.score(x_train, y_train)))

# 0.265



print('{:.3f}'.format(lasso.score(x_test, y_test)))

# 0.214



print(lasso.coef_)

# array([-0.        ,  0.        , -0.        ,  0.        , -0.        ,

#         0.        , -0.        ,  0.        , -0.        , -0.        ,

#        -0.        ,  0.        , -4.38009987, -0.        ,  0.        ,

#        -0.        ,  0.        , -0.        , -0.        , -0.        ,

#        -0.        , -0.        , -0.        , -0.        , -0.        ,

#        -0.        ,  0.        ,  0.        ,  0.        ,  0.        ,

#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,

#         0.        ,  0.        ,  0.        , -0.        ,  0.        ,

#        -0.        , -0.        , -0.        , -0.        , -0.        ,

#        -0.        , -0.        , -0.        , -0.        ,  0.        ,

#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,

#         0.        ,  0.        ,  0.        ,  0.        , -0.        ,

#        -0.        , -0.        , -0.        , -0.        , -0.        ,

#        -0.        , -0.        , -0.        ,  0.        ,  0.        ,

#         0.        , -0.        , -0.        , -0.        ,  0.        ,

#        -0.        , -0.        ,  0.        , -0.        , -0.        ,

#        -4.39984433, -0.        , -0.        ,  0.        , -0.        ,

#        -0.        , -0.        ,  0.        , -0.        , -0.44131553,

#        -0.        , -0.        , -0.        , -0.        , -0.        ,

#        -0.        , -0.        , -0.        , -0.        , -0.        ,

#        -0.        ,  0.        , -0.        , -0.        ])



print(np.sum(lasso.coef_ != 0))

# 3
