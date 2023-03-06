#----------------------------------------------------------------------------------------
# ADA BOOST model churn prediction
#----------------------------------------------------------------------------------------

# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
seed = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
stratify=y,
random_state=seed)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=seed)

# Instantiate an AdaBoost classifier 'adab_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)

# Predict the test set probabilities of positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]

# Evaluate test-set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))


# -------------------------------------------------------------------------------
# Feature importance plot
import pandas as pd
import matplotlib.pyplot as plt

# Create a pd.Series of features importances
importances_adb_clf = pd.Series(adb_clf.feature_importances_, index = X.columns)

# Sort importances_rf
sorted_importances_adb_clf = importances_adb_clf.sort_values(ascending=False,)

# Make a horizontal bar plot
sorted_importances_adb_clf.iloc[:10].plot(kind='barh', color='lightgreen')
plt.show()

# feature importance plot

top_features = sorted_importances_adb_clf.index[:5][::-1]

plt.barh(top_features, sorted_importances_adb_clf.iloc[:5][::-1], color='lightgreen')
plt.title('Top 10 Features Importance for Predicting Churn')
plt.xlabel('Importance')
plt.ylabel('Features')

for index, value in enumerate(sorted_importances_adb_clf.iloc[:5][::-1]):
    plt.text(value, index, str(round(value, 2)))

plt.show()


#----------------------------------------------------------------------------------------
# Plot ROC Curve and AUC Value FOR ADA BOOST

import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# Calculate the predicted probabilities for the model
#y_pred_prob = rf.predict_proba(X_test)[:, 1]

# Calculate the false positive rate and true positive rate for various thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate the area under the ROC curve
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot the ROC curve
sns.set_style('whitegrid')
sns.lineplot(x=fpr, y=tpr)
plt.title('AdaBoost ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Add the AUC to the plot
plt.text(0.7, 0.2, f'AUC: {roc_auc:.2f}', fontsize=12)

# Show the plot
plt.show()
