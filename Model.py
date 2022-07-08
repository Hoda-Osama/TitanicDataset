import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
testid = test['PassengerId']
train_len = len(train)

y = train['Survived']
X = pd.concat([train, test])


def remove_outlier(df_in, col_name):
    mean = df_in[col_name].mean()
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    for ind in range(120):
        if (df_in[col_name].iloc[ind] > fence_high) or (df_in[col_name].iloc[ind] < fence_low):
            df_in[col_name].iloc[ind] = mean

    return df_in[col_name]
X['Title'] = X['Name']

for name_string in X['Name']:
    X['Title'] = X['Name'].str.extract('([A-Za-z]+)\.', expand=True)

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

X.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:
    age_to_impute = X.groupby('Title')['Age'].median()[titles.index(title)]
    X.loc[(X['Age'].isnull()) & (X['Title'] == title), 'Age'] = age_to_impute



X.drop('Title', axis = 1, inplace = True)
X['Family_Size'] = X['Parch'] + X['SibSp']
X['Last_Name'] = X['Name'].apply(lambda x: str.split(x, ",")[0])
X['Fare'].fillna(X['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
X['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in X[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                      'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in X.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 0
X['Fare'].fillna(X['Fare'].median(), inplace = True)

# Making Bins
X['FareBin'] = pd.qcut(X['Fare'], 5)

label = LabelEncoder()
X['FareBin_Code'] = label.fit_transform(X['FareBin'])

X.drop(['Fare'], 1, inplace=True)
X['AgeBin'] = pd.qcut(X['Age'], 4)

label = LabelEncoder()
X['AgeBin_Code'] = label.fit_transform(X['AgeBin'])

X.drop(['Age'], 1, inplace=True)

X['Sex'].replace(['male','female'],[0,1],inplace=True)


X.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked', 'Last_Name', 'FareBin', 'AgeBin', 'Survived'], axis = 1, inplace = True)

X_train = X[:train_len]
X_test = X[train_len:]

y_train = y

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
kfold = StratifiedKFold(n_splits=8)
RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
              "max_features": [3,"sqrt", "log2"],
              "min_samples_split": [2, 4],
              "min_samples_leaf": [5, 7],
              "bootstrap": [False, True],
              "n_estimators" :[200, 500],
              "criterion": ["gini", "entropy"]}

rf_param_grid_best = {"max_depth": [None],
              "max_features": [3],
              "min_samples_split": [4],
              "min_samples_leaf": [5],
              "bootstrap": [False],
              "n_estimators" :[200],
              "criterion": ["gini"]}

gs_rf = GridSearchCV(RFC, param_grid = rf_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)

gs_rf.fit(X_train, y_train) #Score: 0.80143
RFC.fit(X_train, y_train)
rf_best = gs_rf.best_estimator_
print(f'RandomForest GridSearch best params: {gs_rf.best_params_}')
print(f'RandomForest GridSearch best score: {gs_rf.best_score_}')

knn1 = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2,
                           weights='uniform')

knn1.fit(X_train, y_train)
y_pred = gs_rf.predict(X_test)

test_Survived = pd.Series(y_pred, name="Survived")
results = pd.concat([testid,test_Survived],axis=1)
results.to_csv("submit3.csv",index=False) #Score: 0.81578

def svc():
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    test_Survived = pd.Series(y_pred, name="Survived")
    results = pd.concat([testid, test_Survived], axis=1)
    results.to_csv("submitM1.csv", index=False)
def random_forest():
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    test_Survived = pd.Series(y_pred, name="Survived")
    results = pd.concat([testid, test_Survived], axis=1)
    results.to_csv("submitM2.csv", index=False)
def decision_tree():
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    test_Survived = pd.Series(y_pred, name="Survived")
    results = pd.concat([testid, test_Survived], axis=1)
    results.to_csv("submitM3.csv", index=False)



svc() #Score: 0.75598
random_forest() #Score: 0.76315
decision_tree() #Score: 0.76315