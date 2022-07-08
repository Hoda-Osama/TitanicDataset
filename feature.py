import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

def bar_chart_stacked(dataset, feature, stacked = True):
    survived = dataset[dataset['Survived']==1][feature].value_counts()
    dead = dataset[dataset['Survived']==0][feature].value_counts()
    df_survived_dead = pd.DataFrame([survived,dead])
    df_survived_dead.index = ['Survived','Died']
    ax = df_survived_dead.plot(kind='bar',stacked=stacked, figsize=(5,5))
    plt.show()
def bar_chart_compare(dataset, feature1, feature2=None, title = "Survival rate by sex and class'"):
    plt.figure(figsize = [5,5])
    plt.title(title)
    g = sns.barplot(x=feature1, y='Survived', hue=feature2, ci=None, data=dataset).set_ylabel('Survival rate')
def plot_distribution(dataset, feature, title, bins = 30, hist = True, fsize = (5,5)):
    fig, ax = plt.subplots(figsize=fsize)
    ax.set_title(title)
    sns.distplot(train_df[feature], color='b', bins=bins, ax=ax)
    plt.show()
def plot_kernel_density_estimate_survivors(dataset, feature1, title, fsize = (5,5)):
    fig, ax = plt.subplots(figsize=fsize)
    ax.set_title(title)
    sns.kdeplot(dataset[feature1].loc[train_df["Survived"] == 1],
                shade= True, ax=ax, label='Survived').set_xlabel(feature1)
    sns.kdeplot(dataset[feature1].loc[train_df["Survived"] == 0],
                shade=True, ax=ax, label="Died")
def plot_swarm_survivors(dataset, feature1, feature2, title, fize = (155)):
    fig, ax = plt.subplots(figsize=(18,5))
    # Turns off grid on the left Axis.
    ax.grid(True)
    plt.xticks(list(range(0,100,2)))
    sns.swarmplot(y=feature1, x=feature2, hue='Survived',data=train_df).set_title(title)
def show_countplot(dataset, feature, title, fsize = (5,5)):
    fig, ax = plt.subplots(figsize=fsize)
    sns.countplot(dataset[feature], ax=ax).set_title(title)
    plt.show()
def show_compare_countplot(dataset, feature1, feature2, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    p = sns.countplot(x=feature1, hue=feature2, data=dataset, ax=ax).set_title(title)
    plt.show()
#Feature Survived:
bar_chart_stacked(train_df, "Survived")
#Feature Sex:
bar_chart_stacked(train_df, "Sex")
#Feature Pclass
bar_chart_stacked(train_df, 'Pclass')
bar_chart_compare(train_df, "Pclass", "Sex")
#Feature age:
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()
plot_distribution(train_df, "Age", "Age Distribution Passengers")
plot_kernel_density_estimate_survivors(train_df, "Age", "Age Distribution Surived vs Died")
#Features Age and gender together
plot_swarm_survivors(train_df, "Age", "Sex", "Survivor Swarmplot for Age and Gender")
#Features Age and Pclass together
plot_swarm_survivors(train_df, "Age", "Pclass", "Survivor Swarmplot for Age and Pclass")
#Feature Fare
plot_distribution(train_df, "Fare", "Fare Distribution Passengers")
plot_swarm_survivors(train_df, "Fare", "Sex","Survivor Swarmplot for Age and Gender")
#Feature Embarked
bar_chart_stacked(train_df, 'Embarked')
#Feature SibSp
show_compare_countplot(train_df, "SibSp", "Survived", "Survivor count by number of siblings / spouses aboard the Titanic")
#Feature Parch
show_compare_countplot(train_df, "Parch", "Survived", "Survivor count by Parch")


