# titanic
titanic in traditional machine learning


## Model Selection
```
Model on test:
Wide and Deep > Xgboost > Random Forest > Logistic > SVM
0.93            0.89      0.88            0.88        0.80

```


### Pipeline :
0. Clear and definite the problem.(input and output).
1. Load data. Build DataFactory(Dataset).
2. Preview data
3. Drop unrelated column.
4. Fill NA
  a. by other column logically.
  b. using median or mean.
  c. using random forest to predict.
5. Denoise
  a. using IsolationForest(sklearn).
6. Select Model
  a. Using GridSearchCV, select best models and best params.
7. Apply on test samples and score.
