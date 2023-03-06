# -------------------------------------------------------------------------------
# neural net churn prediction
# -------------------------------------------------------------------------------

# import dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# setup and preview the dataframe
df = pd.read_csv('data.csv')

df.shape
df.columns
df.dtypes
df.info()
df.describe()
df.isnull().sum()
df.head()

# convert categorical variables 
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

# build function to create model v01
def create_model():
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=len(X.columns)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# wrap the Keras model inside a KerasClassifier
estimator_200 = KerasClassifier(build_fn=create_model, epochs=200, batch_size=32, verbose=1)

# perform cross-validation
scores_cv05 = cross_val_score(estimator_200, X, y, cv=5)
scores_cv10 = cross_val_score(estimator_200, X, y, cv=10)

print('Cross-validation scores: cv 5', scores_cv05)
print('Cross-validation scores: cv 10', scores_cv10)
print('Mean accuracy: cv 5', scores_cv05.mean())
print('Mean accuracy: cv 10', scores_cv10.mean())

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit the model
estimator_200.fit(X_train, y_train)

# make predictions on the test set
y_pred = estimator_200.predict(X_test)

# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
sns.set_style("darkgrid")
sns.lineplot(x=fpr, y=tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Net ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.show()

# clear plot
plt.clf()


#--------------------------------------------------------------------------------------
# ROC Curve and AUC plot

# fit model and make predictions
model = create_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Confusion Matrix###################################################
# fit model and make predictions
model = create_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# feature importance ##############################################
# fit model and get feature importance
model = create_model()
model.fit(X_train, y_train)
feature_importance = model.feature_importances_

# create feature importance dataframe
df_feature_importance = pd.DataFrame({'Features': X.columns, 'Importance': feature_importance})

# create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_feature_importance.pivot("Features", "Importance"), cmap='Blues', annot=True, fmt=".2f")
plt.title('Feature Importance')
plt.show()
