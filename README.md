# titanic
titanic in traditional machine learning


## Model Selection
```
Model on test:
Wide and Deep > Xgboost > Random Forest > Logistic > SVM
0.80            0.79      0.7655            0.76        0.75

```


### Pipeline :
0. **Clear and definite the problem.(Input and Output).**
1. **Load data. Build DataFactory(Dataset).**
2. **Drop unrelated column.**
3. **Preview data**
4. **Map Categorical Value.**
5. **Fill NA.**  
  - by other column logically.
  - using median or mean.
  - using random forest to predict.
6. **Denoise(Remove Outlier)**
  - using IsolationForest(sklearn).
7. **Composite more features.**
8. **Select Model**
  - Using GridSearchCV, select best models and best params.
9. **Apply on test samples and score.**
10. **Save model and output.**
