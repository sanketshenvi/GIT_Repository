
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[57]:


df=pd.read_csv("C:/Users/Sanket.Shenvi/Desktop/Data Analytics/Deep_Learning/Mobile/train.csv")
dt=pd.read_csv("C:/Users/Sanket.Shenvi/Desktop/Data Analytics/Deep_Learning/Mobile/test.csv")
df.head()


# In[25]:


df.isnull().sum()


# In[26]:


df["price_range"].describe(), df['price_range'].unique()


# In[27]:


corrmat=df.corr()
f,ax=plt.subplots(figsize=(12,10))
sns.heatmap(corrmat,vmax=0.8,square=True,annot=True,annot_kws={'size':8})


# In[49]:


sns.swarmplot(x='battery_power',y='ram',data=df,hue='price_range')
plt.show()


# In[29]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,StratifiedKFold,LeaveOneOut
X_t=df
X_t=df.drop(['price_range'],axis=1)
y_t=df['price_range']
X_t = np.array(X_t)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_t = scaler.fit_transform(X_t)
X_train,X_test,Y_train,Y_test = train_test_split(X_t,y_t,test_size=.20,random_state=42)


# In[30]:


for i in [1,3,5,10,40,60,80,100]:
    clf = SVC(kernel='rbf',C=i).fit(X_train,Y_train)
    scoretrain = clf.score(X_train,Y_train)
    scoretest  = clf.score(X_test,Y_test)
    print("Linear SVM value of i:{}, training score :{:2f} , Test Score: {:2f} \n".format(i,scoretrain,scoretest))


# In[9]:


from sklearn.model_selection import GridSearchCV
clf=SVC(kernel='rbf',C=20)
param_grid = {'C': [1,5,7,10,15,25,50],
              'gamma': [.1,.5,.10,.25,.50,1]}
clf1 = GridSearchCV(clf,param_grid,cv=5)

clf1.fit(X_train,Y_train)
scores = cross_val_score(clf1,X_train,Y_train,cv=5)
skf = StratifiedKFold(5,random_state=10,shuffle=True)
cross_val_score(clf1,X_train,Y_train,cv=skf)


# In[31]:


from sklearn.dummy import DummyClassifier

for strat in ['stratified', 'most_frequent', 'prior', 'uniform']:
    dummy_maj = DummyClassifier(strategy=strat).fit(X_train,Y_train)
    print("Train Stratergy :{} \n Score :{:.2f}".format(strat,dummy_maj.score(X_train,Y_train)))
    print("Test Stratergy :{} \n Score :{:.2f}".format(strat,dummy_maj.score(X_test,Y_test)))


# In[32]:


X = np.array(df.iloc[:,[0,13]])
y = np.array(df['price_range'])
print("Shape of X:"+str(X.shape))
print("Shape of y:"+str(y.shape))
X = scaler.fit_transform(X)


# In[33]:


from matplotlib.colors import ListedColormap
cm_dark = ListedColormap(['#ff6060', '#8282ff','#ffaa00','#fff244','#4df9b9','#76e8fc','#3ad628'])
cm_bright = ListedColormap(['#ffafaf', '#c6c6ff','#ffaa00','#ffe2a8','#bfffe7','#c9f7ff','#9eff93'])
plt.scatter(X[:,0],X[:,1],c=y,cmap=cm_dark,s=10,label=y)
plt.show()


# In[34]:


df.iloc[:,[0,13]].head()


# In[76]:


h = .02  # step size in the mesh
C_param = 1
clf1 = SVC(kernel='linear',C=C_param)
clf1.fit(X, y)
x_min, x_max = X[:, 0].min()-.20, X[:, 0].max()+.20
y_min, y_max = X[:, 1].min()-.20, X[:, 1].max()+.20
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cm_bright)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("SVM Linear Classification (kernal = linear, Gamma = '%s')"% (C_param))
plt.show()


# In[36]:


print("The score of the above :"+str(clf1.score(X,y)))


# In[73]:


A = np.array(dt.iloc[np.arange(1,500),[0,13]])
A = scaler.fit_transform(A)
#X=dt.drop(['price_range'],axis=1)
#y=np.array(dt['price_range'])
#X = np.array(X_t)


# In[75]:


B=clf1.predict(A)
h = .02  # step size in the mesh
C_param = 1
clf1 = SVC(kernel='linear',C=C_param)
clf1.fit(X, y)
x_min, x_max = X[:, 0].min()-.20, X[:, 0].max()+.20
y_min, y_max = X[:, 1].min()-.20, X[:, 1].max()+.20
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cm_bright)
plt.scatter(A[:, 0], A[:, 1], c=B, cmap=cm_dark,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("SVM Linear Classification (kernal = linear, Gamma = '%s')"% (C_param))
plt.show()


# In[74]:


A.shape

