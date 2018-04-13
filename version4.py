import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

'''
得到5个模型作为stacking进行预测
'''
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold

'''
加载训练和测试数据集
'''
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
'''
存储乘客的ID号
'''
PassengerId = test['PassengerId']
# print(train.head(3))
full_data = [train, test]
'''
(1)加入我们自己的特征
(2)给出名字的长度
'''
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
'''
特征表示乘客在Titanic是否有救生艇
'''
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
'''
创造新的家庭成员特征作为SibSp和Parch的组合
'''
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

'''
从新的特征FamillySize,创造新特征（是否一个人）
新增知识： FamillySize 是否一个人分的比较粗，可以在分的细一点
'''
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

'''
消除登船地点缺失的数据
'''
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

'''
去除费用特征缺失数据,并且以他们中位数代替,创建新的特征(费用类别)
'''
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)


def ticketindex(term, ticketsCount):
    return ticketsCount[term]
for dataset in full_data:
    ticketsCount = dataset['Ticket'].value_counts()
    dataset['Ticket'] = dataset['Ticket'].apply(lambda x: ticketindex(x, ticketsCount))
    dataset['TicketIndex'] = 0
    dataset.loc[(dataset['Ticket'] > 1) & (dataset['Ticket'] < 5), 'TicketIndex'] = 1
    dataset.loc[dataset['Ticket'] >= 5, 'TicketIndex'] = 2

'''
创建新的年龄分类特征

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
'''


def quadratic_transform(X, degree=2):
    if degree == 1:
        return X

    quadratic_featurizer = PolynomialFeatures(degree=degree)

    return quadratic_featurizer.fit_transform(X)


def extract_age_feature(df):
    dummies_Sex = pd.get_dummies(train['Sex'], prefix='Sex')

    age_df = train[['Age', 'Sex', 'Parch', 'SibSp', 'Pclass']].copy()
    age_df = pd.concat([age_df, dummies_Sex], axis=1)
    age_df = age_df.filter(regex='Age|Sex_.*|Parch|SibSp|Pclass')

    return age_df

age_df = extract_age_feature(train)

train_data = age_df[age_df.Age.notnull()].as_matrix()
X = train_data[:, 1:]
y = train_data[:, 0]

X_quadratic = quadratic_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_quadratic, y, test_size=0.2, random_state=1)

tuned_parameters = {'n_estimators': [500, 700, 1000], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [2, 3, 4]}

clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)

age_rfr = clf.best_estimator_.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics.mean_absolute_error(y_test, y_pred)

predict_data = age_df[age_df.Age.isnull()].as_matrix()
X_quadratic = quadratic_transform(predict_data[:, 1:])

# 对年龄进行预测
predict_age = age_rfr.predict(X_quadratic)
train.loc[train.Age.isnull(), 'Age'] = predict_age

'''
定义消除乘客名字中的特殊字符
'''


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

'''
创建新的名字特征,包含乘客名字主要信息
'''
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

'''
将所有非常见的标题分组成一个单独的“稀有”组
'''
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    '''
    对性别进行绘制
    '''
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    '''
    对Title进行绘制
    '''
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    '''
    对登船地点绘制
    '''
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    '''
    对费用进行绘制
    '''
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    '''
    对年龄进行绘制
    '''
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

'''
除去特征属性标签
'''
drop_elements = ['PassengerId', 'Name', 'Cabin', 'SibSp', 'Ticket', 'Age']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalFare'], axis=1)
test = test.drop(drop_elements, axis=1)

'''
观察进行过特征清洗,筛选过的新特征数据
'''
# print(train.head(10))
'''
让我们生成一些特征的相关图，看看一个特征和另一个特征的相关程度。
为了做到这一点，我们将利用Seaborn绘图软件包，使我们能够非常方便地
绘制皮尔森相关热图，如下所示
'''
colormap = plt.cm.viridis
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
'''
最后，我们生成一些配对图来观察一个特征和另一个特征的数据分布,
我们再次用Seaborn。

g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic', size=1.2,
       diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
g.set(xticklabels=[])
# plt.show()
'''
'''
这些有用的参数稍后会派上用场的
'''
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0          # for reproducibility
NFOLDS = 5    # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)

'''
定义一个类扩展Sklearn分类器
'''


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(range(ntrain))):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    # 'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    # 'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
    # 'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025
    }

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

'''
将训练集,测试集和目标集转化为Numpy数组输入我们的模型
'''
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values  # 创建训练数据集数组
x_test = test.values  # 创建测试集数组

'''
将训练集和测试集送入模型,然后采用交叉验证方式进行预测,
这些预测结果作为二级模型的新特征
'''
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier

print("Training is complete")

rf_features = rf.feature_importances(x_train, y_train)
et_features = et.feature_importances(x_train, y_train)
ada_features = ada.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train, y_train)
cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame({'features': cols,
                                  'Random Forest feature importances': rf_features,
                                  'Extra Trees  feature importances': et_features,
                                  'AdaBoost feature importances': ada_features,
                                  'Gradient Boost feature importances': gb_features
                                 })


# Scatter plot
trace = go.Scatter(
        y=feature_dataframe['Random Forest feature importances'].values,
        x=feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=1,
            size=25,
            # size= feature_dataframe['AdaBoost feature importances'].values,
            # color = np.random.randn(500), #set color equal to a variable
            color=feature_dataframe['Random Forest feature importances'].values,
            colorscale='Portland',
            showscale=True
        ),
        text=feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
                   autosize=True,
                   title='Random Forest Feature Importance',
                   hovermode='closest',
                   # xaxis= dict(
                   # title= 'Pop',
                   # ticklen= 5,
                   # zeroline= False,
                   # gridwidth= 2,
                   # ),
                   yaxis=dict(
                        title='Feature Importance',
                        ticklen=5,
                        gridwidth=2
                   ),
                   showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter2010')

# Scatter plot
trace = go.Scatter(
                  y=feature_dataframe['Extra Trees  feature importances'].values,
                  x=feature_dataframe['features'].values,
                  mode='markers',
                  marker=dict(
                              sizemode='diameter',
                              sizeref=1,
                              size=25,
                              # size= feature_dataframe['AdaBoost feature importances'].values,
                              # color = np.random.randn(500), #set color equal to a variable
                              color=feature_dataframe['Extra Trees  feature importances'].values,
                              colorscale='Portland',
                              showscale=True
                              ),
                  text=feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
                  autosize=True,
                  title='Extra Trees Feature Importance',
                  hovermode='closest',
                  # xaxis= dict(
                  # title= 'Pop',
                  # ticklen= 5,
                  # zeroline= False,
                  # gridwidth= 2,
                  # ),
                  yaxis=dict(
                    title='Feature Importance',
                    ticklen=5,
                    gridwidth=2
                  ),
                  showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter2010')

# Scatter plot
trace = go.Scatter(
                  y=feature_dataframe['AdaBoost feature importances'].values,
                  x=feature_dataframe['features'].values,
                  mode='markers',
                  marker=dict(
                        sizemode='diameter',
                        sizeref=1,
                        size=25,
                        # size= feature_dataframe['AdaBoost feature importances'].values,
                        # color = np.random.randn(500), #set color equal to a variable
                        color=feature_dataframe['AdaBoost feature importances'].values,
                        colorscale='Portland',
                        showscale=True
                  ),
                  text=feature_dataframe['features'].values
               )
data = [trace]

layout = go.Layout(
              autosize=True,
              title='AdaBoost Feature Importance',
              hovermode='closest',
              # xaxis= dict(
              # title= 'Pop',
              # ticklen= 5,
              # zeroline= False,
              # gridwidth= 2,
              # ),
              yaxis=dict(
                title='Feature Importance',
                ticklen=5,
                gridwidth=2
              ),
              showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter2010')

# Scatter plot
trace = go.Scatter(
    y=feature_dataframe['Gradient Boost feature importances'].values,
    x=feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=1,
        size=25,
        # size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text=feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title='Gradient Boosting Feature Importance',
    hovermode='closest',
    # xaxis= dict(
    # title='Pop',
    # ticklen=5,
    # zeroline=False,
    # gridwidth=2,
    # ),
    yaxis=dict(
        title='Feature Importance',
        ticklen=5,
        gridwidth=2
    ),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter2010')

base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                       'ExtraTrees': et_oof_train.ravel(),
                                       'AdaBoost': ada_oof_train.ravel(),
                                       'GradientBoost': gb_oof_train.ravel()
                                     })
base_predictions_train.head()

'''
二级训练集相关热图
'''
data = [
    go.Heatmap(
        z=base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y=base_predictions_train.columns.values,
        colorscale='Viridis',
        showscale=True,
        reversescale=True
    )
]
py.iplot(data, filename='labelled-heatmap')
'''
一级模型训练和测试预测数据集
作为二次模型的训练和测试集，
然后我们可以拟合二级学习模型了。
'''
x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
'''
只需简单的运行模型中使用的XGBoost参数：

**max_depth** :你想要增长你的树有多深。 如果设置得太高，请注意，可能会有过度拟合的风险。

**gamma** : 在树的叶节点上进一步分区所需的最小损耗减少。 越大，算法越保守。

**eta** : 在每个增压步骤中使用的步骤尺寸缩小以防止过度拟合
'''

gbm = xgb.XGBClassifier(
    # learning_rate = 0.02,
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    # gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(x_train, y_train)
'''
gbm = lgb.LGBMClassifier(
            task='train',
            boosting_type='gbdt',
            objective='binary',
            metric={'l2', 'auc'},
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=0
).fit(x_train, y_train)
'''

predictions = gbm.predict(x_test)


'''
最后，我们已经训练和适应了我们所有的一级和二级模型，
我们现在可以将预测输出到适用于Titanic比赛的格式如下：
生成提交文件
'''

StackingSubmission = pd.DataFrame({'PassengerId': PassengerId,
                            'Survived': predictions})
StackingSubmission.to_csv("StackingSubmission.csv", index=False)
