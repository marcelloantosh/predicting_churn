# -------------------------------------------------------------------------------
# Random forest model churn prediction
# -------------------------------------------------------------------------------

# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# load data
df = pd.read_csv('data.csv')

# preprocess data
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

# Set seed for reproducibility
seed = 1

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = seed)

# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor( n_estimators=400,
                            min_samples_leaf=0.12,
                            random_state=seed)

# Fit 'rf' to the training set
rf.fit(X_train, y_train)

# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# -------------------------------------------------------------------------------
# Feature importance plot
import pandas as pd
import matplotlib.pyplot as plt

# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)

# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values(ascending=False,)

# Make a horizontal bar plot
sorted_importances_rf.iloc[:10].plot(kind='barh', color='lightgreen')
plt.show()

# feature importance plot

top_features = sorted_importances_rf.index[:5][::-1]

plt.barh(top_features, sorted_importances_rf.iloc[:5][::-1], color='lightgreen')
plt.title('Top 10 Features Importance for Predicting Churn')
plt.xlabel('Importance')
plt.ylabel('Features')

for index, value in enumerate(sorted_importances_rf.iloc[:5][::-1]):
    plt.text(value, index, str(round(value, 2)))

plt.show()

# -------------------------------------------------------------------------------
# ROC AUC plot

# Import models and functions
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
seed = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
stratify=y,
random_state=seed)

# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor( n_estimators=400,
                            min_samples_leaf=0.12,
                            random_state=seed)

# Fit 'rf' to the training set
rf.fit(X_train, y_train)

# Predict the test set labels 'y_pred'
y_pred_rf = rf.predict(X_test)

# Evaluate test-set roc_auc_score
rf_roc_auc_score = roc_auc_score(y_test, y_pred_rf)

# Print rf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(rf_roc_auc_score))

# Plot ROC Curve and AUC Value FOR ADA BOOST
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# Calculate the false positive rate and true positive rate for various thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)

# Calculate the area under the ROC curve
roc_auc = roc_auc_score(y_test, y_pred_rf)

# Plot the ROC curve
sns.set_style('whitegrid')
sns.lineplot(x=fpr, y=tpr)
plt.title('Random Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Add the AUC to the plot
plt.text(0.7, 0.2, f'AUC: {roc_auc:.2f}', fontsize=12)

# Show the plot
plt.show()





















#------------------------------------------------------------------------------------

# import dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

# load data
df = pd.read_csv('data.csv')

# preprocess data
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# train random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=3)
rf.fit(X_train, y_train)

# evaluate performance on test set
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# perform cross-validation
scores_cv05 = cross_val_score(rf, X, y, cv=5)
scores_cv10 = cross_val_score(rf, X, y, cv=10)

print('Cross-validation scores: cv 5', scores_cv05)
print('Cross-validation scores: cv 10', scores_cv10)
print('Mean accuracy: cv 5', scores_cv05.mean())
print('Mean accuracy: cv 10', scores_cv10.mean())

# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
sns.lineplot(x=fpr, y=tpr)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
sns.set_style("darkgrid")
sns.lineplot(x=fpr, y=tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.show()


# clear plot
plt.clf()


# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

# plot feature importances
feature_importances = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
sns.barplot(x='importance', y=feature_importances.index, data=feature_importances)


