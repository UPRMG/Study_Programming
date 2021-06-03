#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pkg_resources 
import pandas 
OutputDataSet = pandas.DataFrame(sorted([(i.key, i.version) for i in pkg_resources.working_set])) 
print(OutputDataSet)


# # 모델링
#     1. 데이터 파악 
#     2. 데이터 전처리 (결측값, 명목변수 처리, 스케일링)
#     3. 데이터 분리(train_test_split)
#     4. 변수 선택 (Feature Selection)
#     5. 모델 학습 (model_selection : cross_val, grid)
#     6. 다양한 모델
#     7. 성능 평가
# ---------
# # 웹크롤링 (request, get, url)
# # 자연어처리 (split, tf-idf)

# ---

# # 모델링
# 1. 데이터 파악

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('./data/check')


# 2. 데이터 전처리(결측값, 명목변수 처리, 스케일링)

# In[77]:


data.isnull().sum()


# In[78]:


data.dropna()


# In[79]:


data.fillna(0)


# In[81]:


data.fillna(method='ffill')


# In[82]:


data.fillna({0:data[0].mean()})


# In[127]:


data = ['g','t','tg','d','d'] # Test 데이터 모두 합하여 진행 후 분리


# In[123]:


one_hot_data = pd.get_dummies(data)


# In[124]:


one_hot_data


# In[128]:


from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# In[149]:


data = pd.DataFrame(np.array([1,2,3,4,5]))


# In[150]:


data


# In[151]:


data = np.array([[1],[2],[3],[4],[5]])


# In[155]:


standardScaler_s = StandardScaler()

standardScaler_s.fit(data)

raw_df = standardScaler_s.transform(data)


# In[156]:


RobustScaler_r = RobustScaler()

RobustScaler_r.fit(data)

raw_df = RobustScaler_r.transform(data)


# In[157]:


MinMaxScaler_r = MinMaxScaler()

MinMaxScaler_r.fit(data)

raw_df = MinMaxScaler_r.transform(data)


# 3. 데이터 분리(train_test_split)

# In[159]:


import sklearn
from sklearn.model_selection import train_test_split


# In[160]:


x_data = pd.DataFrame(np.array([[1,2],[3,4],[5,6],[7,8]]))


# In[161]:


y_data = pd.DataFrame(np.array([1,2,3,4]))


# In[162]:


x_data


# In[163]:


y_data


# In[164]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                    shuffle=False, 
                                   test_size=0.2)


# 4. 변수 선택(Feature Selection)

# In[175]:


pd.DataFrame(X).corr()


# In[166]:


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
print(selector.support_)

print(selector.ranking_)


# 5. 모델 학습

# In[183]:


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


# In[199]:


iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(iris.data, iris.target)

sorted(clf.cv_results_.keys())


# In[200]:


clf.score(iris.data, iris.target)


# In[201]:


clf.best_params_


# In[185]:


import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance, plot_tree


# In[202]:


iris = datasets.load_iris()
parameters = {'max_depth':(5,10,15), 'learning_rate':[0.01, 0.1]}
svc = XGBClassifier()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(iris.data, iris.target)

sorted(clf.cv_results_.keys())


# In[203]:


clf.score(iris.data, iris.target)


# In[204]:


clf.best_params_


# In[222]:


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()


# In[223]:


cv_results = cross_validate(lasso, X, y, cv=3)
sorted(cv_results.keys())

cv_results['test_score']


# In[224]:


scores = cross_validate(lasso, X, y, cv=3,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
print(scores['test_neg_mean_squared_error'])

print(scores['train_r2'])
print(scores['test_r2'])


# In[279]:


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = LinearRegression()


# In[280]:


cv_results = cross_validate(lasso, X, y, cv=3)
sorted(cv_results.keys())

cv_results['test_score']


# In[281]:


scores = cross_validate(lasso, X, y, cv=3,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
print(scores['test_neg_mean_squared_error'])

print(scores['train_r2'])
print(scores['test_r2'])


# ### 성능 향상 시도 진행 (스케일링, 변수선택 등)

# 6. 다양한 모델

# In[274]:


from sklearn.datasets import load_iris,load_wine,load_breast_cancer,load_boston,load_diabetes,load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import *

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA
from sklearn.decomposition import TruncatedSVD, DictionaryLearning, FactorAnalysis
from sklearn.decomposition import FastICA, NMF, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[ ]:


base_model = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier())

bagging_model = BaggingClassifier(base_model, n_estimators=10, max_samples=0.5, max_features=0.5)


# In[ ]:


cross_val = cross_validate(
    estimator=bagging_model,
    X=cancer.data, y=cancer.target,
    cv=5)

print('avg fit time : {} (+/- {})'.format(cross_val['fit_time'].mean(), cross_val['fit_time'].std()))
print('avg fit time : {} (+/- {})'.format(cross_val['score_time'].mean(), cross_val['score_time'].std()))
print('avg fit time : {} (+/- {})'.format(cross_val['test_score'].mean(), cross_val['test_score'].std()))


# In[ ]:


cross_val = cross_validate(
    estimator=base_model,
    X=cancer.data, y=cancer.target,
    cv=5)

print('avg fit time : {} (+/- {})'.format(cross_val['fit_time'].mean(), cross_val['fit_time'].std()))
print('avg fit time : {} (+/- {})'.format(cross_val['score_time'].mean(), cross_val['score_time'].std()))
print('avg fit time : {} (+/- {})'.format(cross_val['test_score'].mean(), cross_val['test_score'].std()))


# In[ ]:


model = PCA(n_components=3, random_state=0)
model.fit(df)
transformed_df = model.transform(df)
transformed_df.shape


# 7. 성능평가
# 

# In[294]:


from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[304]:


import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)


# In[305]:


plt.plot(fpr,tpr)


# In[297]:


from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_squared_error(y_true, y_pred))

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_squared_error(y_true, y_pred, squared=False))

y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]
print(mean_squared_error(y_true, y_pred))

print(mean_squared_error(y_true, y_pred, squared=False))
print(mean_squared_error(y_true, y_pred, squared=True))

print(mean_squared_error(y_true, y_pred, multioutput='raw_values'))

print(mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7]))


# In[298]:


from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)


# In[300]:


import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
print(fpr)

print(tpr)

print(thresholds)


# In[303]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(solver="liblinear").fit(X, y)
roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')


# # 웹크롤링 (request, get, url)

# In[307]:


from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
import requests


# In[309]:


url_query = quote(str(input('검색어 :')))
url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query='+url_query

j = int(input('원하는 검색 페이지수 :'))
print()

df_title_list = []
df_company_list = []
df_url_list = []

for i in range(j):
    
    search_url = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(search_url, 'html.parser')

    check = soup.find_all('div', {'class':'news_area'})

    for i in range(len(check)):
        a = check[i].find('a',{'class':'news_tit'})['title']
        b = check[i].find('a',{'class':'info press'}).text
        c = check[i].find('a',{'class':'news_tit'})['href']
#         print('news_title = ', a)
#         print('news_compant = ', b)
#         print('news_url = ', c)
        
        df_title_list.append(a)
        df_company_list.append(b)
        df_url_list.append(c)
        
    try:
        ab = soup.find('a',{'class':'btn_next'}).get('href')
        url = 'https://search.naver.com/search.naver' + ab
    except:
        break
    
#     print()
    
news_df_frame = pd.DataFrame([df_company_list, df_title_list, df_url_list],index=['company', 'title', 'url'])
news_df_frame = news_df_frame.T

news_df_frame


# # 자연어처리 (split, tf-idf)

# In[315]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[316]:


from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.


# In[317]:


from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)


# # 기타(Scipy)

# In[321]:


from scipy.optimize import fmin_bfgs
import scipy


# In[311]:


def f(x):
    return x**2 + 10*np.sin(x)

x = np.arange(-10, 10, 0.1)
plt.plot(x, f(x))
plt.show()


# In[312]:


fmin_bfgs( f, 0 )


# In[314]:


fmin_bfgs( f, 5 )


# # Matplotlib

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib as mpl
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


plt.plot(np.random.randn(50),label='a')
plt.plot(np.random.randn(50),label='b')
plt.plot(np.random.randn(50),label='c')
plt.title('title')
plt.xlabel('x')
plt.ylabel('random.randn')
plt.legend()


# In[20]:


height = [np.random.randn() * i for i in range(1,6)]
names = ['a','b','c','d','e']
y_pos = np.arange(len(names))
plt.bar(y_pos,height)
plt.xticks(y_pos,names,fontweight='bold')
plt.xlabel('group')


# In[31]:


# plt.subplots_adjust(wspace=1)

dt=0.01
t = np.arange(0,30,dt)
n1 = np.random.randn(len(t))
n2 = np.random.randn(len(t))
r = np.exp(-t/0.05)

c1 = np.convolve(n1,r,mode='same')*dt
c2 = np.convolve(n2,r,mode='same')*dt

s1 = 0.01*np.sin(2*np.pi*10*t)+c1
s2 = 0.01*np.sin(2*np.pi*10*t)+c2

plt.subplot(211)
plt.plot(t,s1,t,s2)
plt.xlim(0,5)
plt.xlabel('time')
plt.ylabel('s1&s2')
plt.grid(True)

plt.subplot(212)
plt.cohere(s1,s2,256,1./dt)
plt.ylabel('coherernece')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ---

# ---

# ---

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




