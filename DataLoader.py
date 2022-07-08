import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Load dataSet
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
print(train_df.head())
print(test_df.head())
survived = train_df[train_df['Survived']==1]
not_survided = train_df[train_df['Survived']==0]
total_passenger = len(train_df.Survived)
print('How was the percentage of survived?')
print('-'*100)
print('The percentage of survived was: {:.2f}%'.format((len(survived)/total_passenger)*100))
print('The percentage of not survived was:{:.2f}%'.format((len(not_survided)/total_passenger)*100))
sns.countplot(x = 'Survived', data = train_df)
fig, ax = plt.subplots(figsize=(30, 10))
sns.boxplot(data=train_df)
fig, ax=plt.subplots(figsize=(30, 10))
ax = sns.heatmap(train_df.corr(), fmt='.2f', annot=True, ax=ax, vmin=-1, vmax=1 )

#Check missing values
def plot_missing_data(dataset, title):
    fig, ax = plt.subplots(figsize=(5,5))
    plt.title(title)
    sns.heatmap(dataset.isnull(), cbar=False)
    plt.show()

print(train_df.info())
plot_missing_data(train_df, "Training Dataset")
print(test_df.info())
plot_missing_data(test_df, "Test Dataset")



