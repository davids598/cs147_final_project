import cupy as cp

import cudf

# Both import methods supported

from cuml import LinearRegression

from cuml.linear_model import LinearRegression

lr = LinearRegression(fit_intercept = True, normalize = False,

                      algorithm = "eig")

X = cudf.DataFrame()

X['col1'] = cp.array([1,1,2,2], dtype=cp.float32)

X['col2'] = cp.array([1,2,2,3], dtype=cp.float32)

y = cudf.Series(cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32))

reg = lr.fit(X,y)

print(reg.coef_)

print(reg.intercept_)

X_new = cudf.DataFrame()

X_new['col1'] = cp.array([3,2], dtype=cp.float32)

X_new['col2'] = cp.array([5,5], dtype=cp.float32)

preds = lr.predict(X_new)

print(preds)  
