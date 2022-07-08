# TitanicDataset
the Titanic Problem is based on the sinking of the ‘Unsinkable’ ship Titanic in the early 1912. It gives you information about multiple people like their ages, sexes, sibling counts, embarkment points and whether or not they survived the disaster.
![image](https://user-images.githubusercontent.com/63510426/177967387-c83aef64-855d-40ba-aa03-0698a441f703.png)
![image](https://user-images.githubusercontent.com/63510426/177967425-daa70b66-fad2-487a-81a6-2474db7c7f84.png)
![image](https://user-images.githubusercontent.com/63510426/177967443-7e6b0aca-2460-4d15-9471-d2ccca1cc5b4.png)
![image](https://user-images.githubusercontent.com/63510426/177967505-83880a82-5b47-42ba-923e-47baab4d8de4.png)
![image](https://user-images.githubusercontent.com/63510426/177967529-44f668d0-7016-4146-83e1-0f774d21225f.png)
the majority of passengers in the training data died. Only 38% survived the disaster.
![image](https://user-images.githubusercontent.com/63510426/177967616-53a8e127-40d3-44e4-bf70-e6a0128ccb9c.png)
![image](https://user-images.githubusercontent.com/63510426/177967659-1f97d2c0-1d63-4789-ac8f-bb048a710644.png)
![image](https://user-images.githubusercontent.com/63510426/177967679-dc02b45a-4a2f-4300-9d98-d26c0e80ab42.png)
![image](https://user-images.githubusercontent.com/63510426/177967712-a15ab849-6766-45f6-bc0f-0e075ab960cc.png)
even though the majority of the passenger were male, the majority of survivors were female.
![image](https://user-images.githubusercontent.com/63510426/177967753-cadcdb9a-cd9e-4cdc-a297-156259666bc9.png)
![image](https://user-images.githubusercontent.com/63510426/177967787-36c506f2-b476-4d2a-a4d1-2a967ce82cd0.png)
Most passenger had class 3 tickets, yet only 24% of class 3 passengers survived.
Almost 63% of the passenger from class 1 survived.
approx 50% of the class 2 passenger survived.
![image](https://user-images.githubusercontent.com/63510426/177967844-9c47ead2-e99d-4601-b661-33ae24106315.png)
![image](https://user-images.githubusercontent.com/63510426/177967860-48025848-5b13-4d28-834d-55eb130c32f2.png)
Survival Rate females 1. Class: 96,8%
Survival Rate females 2. Class: 92,1%
Survival Rate females 3. Class: 50%
Survival Rate male 1. Class: 36.8%
![image](https://user-images.githubusercontent.com/63510426/177967941-c6f78cc7-96c4-4583-b946-a4c3d8ca96a9.png)
![image](https://user-images.githubusercontent.com/63510426/177967954-9678ef47-db3d-464a-aa56-5ce35716fdb6.png)
![image](https://user-images.githubusercontent.com/63510426/177967970-710b78d4-ca83-47d4-8ed3-2acb9447d08b.png)
The Histogram shows that age follows a fairly normal distribution
![image](https://user-images.githubusercontent.com/63510426/177968032-5a7a18e2-b18c-4289-a7a3-7e9728982a4e.png)
![image](https://user-images.githubusercontent.com/63510426/177968064-3a060d40-732b-450a-9fbd-c9e28d1f865e.png)
![image](https://user-images.githubusercontent.com/63510426/177968085-bcf63092-4727-45b9-9b1a-3cbd3442b61f.png)
![image](https://user-images.githubusercontent.com/63510426/177968100-e977f007-bf1a-4285-88d1-d25e205a9fd3.png)
![image](https://user-images.githubusercontent.com/63510426/177968117-c4a3d091-8601-4811-86f2-138e965101da.png)
![image](https://user-images.githubusercontent.com/63510426/177968139-a0c1e7e9-e610-4015-b524-89757bc99df0.png)
![image](https://user-images.githubusercontent.com/63510426/177968178-15916ec2-b07f-4dc0-9610-efcf4031d7ef.png)
![image](https://user-images.githubusercontent.com/63510426/177968202-b75179dd-be2c-49d4-a27b-625e44a7c9fa.png)
Fare does not follow a normal distribution and has a huge spike at the price range [0-100$].
![image](https://user-images.githubusercontent.com/63510426/177968249-17429599-d49a-4780-9abe-0c70fdf914f5.png)
![image](https://user-images.githubusercontent.com/63510426/177968275-00fc939c-0939-4bfb-abce-a300bbfc0ee4.png)
![image](https://user-images.githubusercontent.com/63510426/177968313-d5cdecd0-d514-4b05-b996-25548a14a14f.png)
![image](https://user-images.githubusercontent.com/63510426/177968355-169de627-eb48-4ca7-8055-796590a048c7.png)
![image](https://user-images.githubusercontent.com/63510426/177968413-e40763fe-db4e-48f4-be8b-7d3639666f39.png)
![image](https://user-images.githubusercontent.com/63510426/177968441-51aea1cd-22f0-4dd6-a202-d440fc9b1267.png)
PROPOSED MODEL
Support Vector Machine
classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    submission.to_csv('submission1_1.csv', index=False)
In this model, we used the RBF kernel and fit the model with the training and testing data. When the model is trained, the output is the testing data the model’s prediction if the passenger either survived or not.

Random Forest
 random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred
        })
    submission.to_csv('submission2_2.csv', index=False)
    
In this model, similar to the SVM model, we fit the training and testing data and the output is the passenger ID and whether that passenger have survived or not

Decision Trees

decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print(acc_decision_tree)
    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred
        })
    submission.to_csv('submission2_3.csv', index=False)
