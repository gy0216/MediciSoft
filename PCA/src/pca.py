### data 불러오기
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
'Alcalinity of ash', 'Magnesium', 'Total phenols', 
'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()


### 데이터 전처리 - 데이터셋 분리
from sklearn.cross_validation import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)
        
        
### 데이터 전처리 - 데이터 표준화 작업
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


### 공분산 행렬을 이용한 Eigendecomposition
import numpy as np

cov_mat = np.cov(X_train_std.T) # 공분산 행렬을 생성해주는 함수
# T는 Matrix의 T를 의미. 함수에 맞는 파라미터로 쓰기 위해 행렬을 돌려줌

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)


### 에이겐벨류의 설명 분산 비율
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 에이겐벨류 / 에이겐벨류의 합 을 각각 구한다. 나온 각각의 값은 아이겐벨류의 설명 분산 비율이다.
# 즉, 어떤 에이겐벨류가 가장 설명력이 높은지를 비율로 나타내기 위한 것이다.

cum_var_exp = np.cumsum(var_exp) # 누적 합을 계산해주는 함수. -> 누적 백분위로 표현


### 에이겐벨류의 영향력을 그래프로 시각화
import matplotlib.pyplot as plt

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/pca1.png', dpi=300)
#plt.show()

### 에이겐 쌍을 이용하여 투영행렬 생성
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
# 에이겐 쌍 생성 -> 투플 자료형

eigen_pairs.sort(reverse=True) # 내림차순으로 정렬

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
# 투영행렬 W : 변수를 2차원으로 축소시키는 투영행렬.
# eigen_pairs의 0,1 번째만 -> 2개의 에이겐 쌍으로만 차원축소를 하겠다는 것.
# hstack -> 행의 수가 같은 두 개 이상의 배열을 옆으로 연결하여, 열의 수가 늘어난 np배열을 만든다.
# 1차원 배열끼리는 hstack 되지 않으므로 [:, np.newaxis]을 추가함.

print('Matrix W:\n', w)


### 투영행렬로 피처 압축
X_train_std[0].dot(w) # X_train_std[0] 행렬과 W 행렬의 곱(내적연산)

X_train_pca = X_train_std.dot(w) # 피처를 투영행렬에 곱한 값 -> 피처 축소된 결과


### 변환된 데이터를 그래프로 시각화
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], 
                X_train_pca[y_train==l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./figures/pca2.png', dpi=300)
plt.show()

