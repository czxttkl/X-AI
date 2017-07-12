import numpy
from sklearn.preprocessing import PolynomialFeatures

k = 3
d = 2
# power of one, two and interaction
kk = k + k + k * (k-1) / 2
kk = int(kk)
coef = numpy.random.rand(kk)
poly = PolynomialFeatures(2, include_bias=False)

one_idx = numpy.random.choice(k, d, replace=False)
feature = numpy.zeros(k)
feature[one_idx] = 1
trans_feature = poly.fit_transform(feature.reshape((1, -1))).reshape((1, -1))

print(coef)
print(feature)
print(trans_feature)
