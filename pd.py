import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay, confusion_matrix
)

data = pd.read_csv('/content/DataPD.txt', sep="\t")
data.head()

desc = pd.read_csv('/content/Description.txt', sep="\t")
print(tabulate(desc, headers='keys', tablefmt='plain'))

variable_types = desc['Type'].unique()
print("Variable types:", variable_types)

# Missing value analysis
data_stats = pd.DataFrame({
    'Name': data.columns,
    'Valid': data.notna().sum().values,
    'Empty': data.isna().sum().values
})

# Remove 'ID' and 'deflag' from analysis
data_stats = data_stats[~data_stats['Name'].isin(['ID', 'deflag'])]

print(tabulate(data_stats, headers='keys', tablefmt='simple'))

# Descriptive stats
average = data.mean(numeric_only=True).reset_index().rename(columns={'index': 'Name', 0: 'Average'})
std_dev = data.std(numeric_only=True).reset_index().rename(columns={'index': 'Name', 0: 'Std_Dev'})
quant = data.quantile(q=np.linspace(0.00, 1.00, 5), numeric_only=True).transpose()
quant.columns = ['Min', 'Q_0.25', 'Q_0.50', 'Q_0.75', 'Max']
quant = quant.reset_index().rename(columns={'index': 'Name'})

data_stats = data_stats.merge(average, on='Name', how='left')
data_stats = data_stats.merge(std_dev, on='Name', how='left')
data_stats = data_stats.merge(quant, on='Name', how='left')

# Plotting missing or extreme values
print(f"==> NA values in var2_AQ: {data['var2_AQ'].isna().sum()}")
print(f"==> Count of var2_AQ >= 5: {(data['var2_AQ'] >= 5).sum()}")

# Histogram for one variable
plt.hist(data['var1_AQ'].dropna(), bins=100)
plt.title("Histogram: var1_AQ")
plt.show()

# Histograms for all numeric variables
numeric_vars = data.select_dtypes(include=[np.number]).columns.drop(['ID', 'deflag'])
fig, axes = plt.subplots(7, 4, figsize=(20, 20))
axes = axes.flatten()

color_list = ['green', 'purple', 'olive', 'red', 'orange']
type_list = desc['Type'].unique()
type_color_dict = dict(zip(type_list, color_list))
print(type_color_dict)

for i, col in enumerate(numeric_vars):
    ax = axes[i]
    values = data[col].dropna()

    # remove outliers
    values = values[(values >= values.quantile(0.01)) & (values <= values.quantile(0.99))]

    var_type = desc[desc['Criteria'] == col]['Type'].values[0]
    color = type_color_dict.get(var_type)

    ax.hist(values, bins=100, density=True, color=color)
    ax.set_title(col)

plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = data[numeric_vars].corr()
print(correlation_matrix)
plt.matshow(correlation_matrix)
plt.title("Correlation Matrix")
plt.colorbar()
plt.show()

#Prepare clean regression dataset
data_regression = data.replace(99.0, np.nan)
data_regression['ID'] = data['ID']
data_regression = data_regression.fillna(data.median(numeric_only=True))

X = data_regression.drop(columns=['ID', 'deflag'])
Y = data_regression['deflag']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, stratify=Y, random_state=285, train_size=0.8
)

print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')
print(f'Default ratio - train: {100*Y_train.mean():.4f}%')
print(f'Default ratio - test:  {100*Y_test.mean():.4f}%')

# Logistic regression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, Y_train)

# ROC Curve
Y_score = lr.decision_function(X_test)
fp_rate, tp_rate, _ = roc_curve(Y_test, Y_score, pos_label=lr.classes_[1])
roc_display = RocCurveDisplay(fpr=fp_rate, tpr=tp_rate)
roc_display.plot()
plt.title("ROC Curve")
plt.show()

# AUC
auc = roc_auc_score(Y_test, Y_score)
print(f"AUC score: {auc:.4f}")

# Confusion matrix
Y_pred = lr.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# PD estimation
for bank_id, name in [(484, 'ABC'), (47, 'XYZ'), (2741, 'QQQ')]:
    row = data_regression[data_regression['ID'] == bank_id].drop(columns=['deflag', 'ID'])
    pd_value = lr.predict_proba(row)[0, 1]
    print(f'PD_{name} = {pd_value * 100:.2f}%')

import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay, confusion_matrix
)

data = pd.read_csv('/content/DataPD.txt', sep="\t")
data.head()

desc = pd.read_csv('/content/Description.txt', sep="\t")
print(tabulate(desc, headers='keys', tablefmt='plain'))

variable_types = desc['Type'].unique()
print("Variable types:", variable_types)

# Missing value analysis
data_stats = pd.DataFrame({
    'Name': data.columns,
    'Valid': data.notna().sum().values,
    'Empty': data.isna().sum().values
})

# Remove 'ID' and 'deflag' from analysis
data_stats = data_stats[~data_stats['Name'].isin(['ID', 'deflag'])]
print(tabulate(data_stats, headers='keys', tablefmt='simple'))

# Descriptive stats
average = data.mean(numeric_only=True).reset_index().rename(columns={'index': 'Name', 0: 'Average'})
std_dev = data.std(numeric_only=True).reset_index().rename(columns={'index': 'Name', 0: 'Std_Dev'})
quant = data.quantile(q=np.linspace(0.00, 1.00, 5), numeric_only=True).transpose()
quant.columns = ['Min', 'Q_0.25', 'Q_0.50', 'Q_0.75', 'Max']
quant = quant.reset_index().rename(columns={'index': 'Name'})

data_stats = data_stats.merge(average, on='Name', how='left')
data_stats = data_stats.merge(std_dev, on='Name', how='left')
data_stats = data_stats.merge(quant, on='Name', how='left')

# Plotting missing or extreme values
print(f"==> NA values in var2_AQ: {data['var2_AQ'].isna().sum()}")
print(f"==> Count of var2_AQ >= 5: {(data['var2_AQ'] >= 5).sum()}")

# Histogram for one variable
plt.hist(data['var1_AQ'].dropna(), bins=100)
plt.title("Histogram: var1_AQ")
plt.show()

# Histograms for all numeric variables
numeric_vars = data.select_dtypes(include=[np.number]).columns.drop(['ID', 'deflag'])
fig, axes = plt.subplots(7, 4, figsize=(20, 20))
axes = axes.flatten()

color_list = ['green', 'purple', 'olive', 'red', 'orange']
type_list = desc['Type'].unique()
type_color_dict = dict(zip(type_list, color_list))
print(type_color_dict)

for i, col in enumerate(numeric_vars):
    ax = axes[i]
    values = data[col].dropna()

    # remove outliers
    values = values[(values >= values.quantile(0.01)) & (values <= values.quantile(0.99))]

    var_type = desc[desc['Criteria'] == col]['Type'].values[0]
    color = type_color_dict.get(var_type)

    ax.hist(values, bins=100, density=True, color=color)
    ax.set_title(col)

plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = data[numeric_vars].corr()
print(correlation_matrix)
plt.matshow(correlation_matrix)
plt.title("Correlation Matrix")
plt.colorbar()
plt.show()

#Prepare clean regression dataset
data_regression = data.replace(99.0, np.nan)
data_regression['ID'] = data['ID']
data_regression = data_regression.fillna(data.median(numeric_only=True))

X = data_regression.drop(columns=['ID', 'deflag'])
Y = data_regression['deflag']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, stratify=Y, random_state=285, train_size=0.8
)

print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')
print(f'Default ratio - train: {100*Y_train.mean():.4f}%')
print(f'Default ratio - test:  {100*Y_test.mean():.4f}%')

# Logistic regression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, Y_train)

# ROC Curve
Y_score = lr.decision_function(X_test)
fp_rate, tp_rate, _ = roc_curve(Y_test, Y_score, pos_label=lr.classes_[1])
roc_display = RocCurveDisplay(fpr=fp_rate, tpr=tp_rate)
roc_display.plot()
plt.title("ROC Curve")
plt.show()

# AUC
auc = roc_auc_score(Y_test, Y_score)
print(f"AUC score: {auc:.4f}")

# Confusion matrix
Y_pred = lr.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# PD estimation
for bank_id, name in [(484, 'ABC'), (47, 'XYZ'), (2741, 'QQQ')]:
    row = data_regression[data_regression['ID'] == bank_id].drop(columns=['deflag', 'ID'])
    print(lr.predict_proba(row))
    pd_value = lr.predict_proba(row)[0, 1]
    print(f'PD_{name} = {pd_value * 100:.2f}%')