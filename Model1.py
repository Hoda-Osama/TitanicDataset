import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
#drop unneeded values in train
train_df=train_df.drop("PassengerId",axis=1)
train_df=train_df.drop("Name",axis=1)
train_df=train_df.drop("Ticket",axis=1)
train_df=train_df.drop("Cabin",axis=1)
#drop unneeded values in test
test_df=test_df.drop("Name",axis=1)
test_df=test_df.drop("Ticket",axis=1)
test_df=test_df.drop("Cabin",axis=1)

data = [train_df, test_df]
#age feature
for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
print(train_df["Age"].isnull().sum())
print("age")

#embarked feature
common_value = 'S'
train_df["Embarked"] = train_df["Embarked"].fillna(common_value)
#train_df = train_df.fillna(test_df['Fare'].mean())
test_df = test_df.fillna(test_df['Fare'].mean())
print(train_df.info())

#Encoding categorical data

le = LabelEncoder()
train_df["Sex"]= le.fit_transform(train_df["Sex"])
print(train_df["Sex"])
le = LabelEncoder()
test_df["Sex"]= le.fit_transform(test_df["Sex"])
print(test_df["Sex"])
le = LabelEncoder()
train_df["Embarked"]= le.fit_transform(train_df["Embarked"])
print(train_df["Embarked"])
le = LabelEncoder()
test_df["Embarked"]= le.fit_transform(test_df["Embarked"])
print(test_df["Embarked"])

#Spliting the Train & Test datasets
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()
print(train_df.head(10))
print(test_df.head(10))

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train)
X_test = sc.transform(X_test)


def svc():
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred
        })
    submission.to_csv('submission1_1.csv', index=False)
#Your submission scored 0.77990'''

def random_forest():
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred
        })
    submission.to_csv('submission1_2.csv', index=False)
def decision_tree():
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred
        })
    submission.to_csv('submission1_3.csv', index=False)



svc() #Score: 0.78468
random_forest() #Score: 0.72009
decision_tree() #Score: 0.69138
