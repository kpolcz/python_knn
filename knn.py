from collections import Counter

import numpy as np
import pandas as pd
import scipy as sc
import math
from sklearn import datasets



# learning_dataset = pd.read_csv('iris.data.learning')
# test_dataset = pd.read_csv('iris.data.test')
##learning_dataset.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
# test_dataset.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
# print(learning_dataset.head(10))


class KNN:

    def __init__(self, learning_dataset, k):
        self.learning = pd.read_csv(learning_dataset)
        self.k = k

    def predict(self,  to_predict, k=3):

        distributions = []
        for group in self.learning:
            for features in self.learning[group]:
                euc_dist = np.linalg.norm(np.array(features) - np.array(to_predict))
                distributions.append([euc_dist, group])

            results = [i[1] for i in sorted(distributions)[:k]]

            result = Counter(results).most_common(1)[0][0]

            confidence = Counter(results).most_common(1)[0][1] / k
        return result


kn = KNN('iris.data.learning', 3)
iris = datasets.load_iris()
print(iris)
#kn.predict()