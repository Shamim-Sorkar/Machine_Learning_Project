import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Step 1: Data Preprocessing
data = pd.read_csv("dataset.csv")  # Replace "dataset.csv" with the actual file path
data.dropna(inplace=True)
X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to evaluate the model and return the metrics
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Initialize precision and recall to zero
    f1 = precision = recall = 1

    if conf_matrix.shape[0] < 2 or conf_matrix.shape[1] < 2:
        # Handle the case when the confusion matrix has only one class
        tp = fp = fn = tn = 0
    else:
        # Assign TP, FP, FN, TN from confusion matrix
        tp = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        tn = conf_matrix[1, 1]

        # Calculate precision and recall
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1, conf_matrix

# Dictionary to store results
results = {}
confusion_matrices = {}

# Model Training - Logistic Regression
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
log_reg_results_train = evaluate_model(log_reg_model, X_train, y_train)
log_reg_results_test = evaluate_model(log_reg_model, X_test, y_test)
results['Logistic Regression'] = log_reg_results_test[:-1]
confusion_matrices['Logistic Regression'] = log_reg_results_test[-1]

# Model Training - Decision Tree
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
decision_tree_results_train = evaluate_model(decision_tree_model, X_train, y_train)
decision_tree_results_test = evaluate_model(decision_tree_model, X_test, y_test)
results['Decision Tree'] = decision_tree_results_test[:-1]
confusion_matrices['Decision Tree'] = decision_tree_results_test[-1]

# Model Training - Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
random_forest_results_train = evaluate_model(random_forest_model, X_train, y_train)
random_forest_results_test = evaluate_model(random_forest_model, X_test, y_test)
results['Random Forest'] = random_forest_results_test[:-1]
confusion_matrices['Random Forest'] = random_forest_results_test[-1]

# Model Training - Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_results_train = evaluate_model(svm_model, X_train, y_train)
svm_results_test = evaluate_model(svm_model, X_test, y_test)
results['SVM'] = svm_results_test[:-1]
confusion_matrices['SVM'] = svm_results_test[-1]

# Model Training - k-Nearest Neighbors (kNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_results_train = evaluate_model(knn_model, X_train, y_train)
knn_results_test = evaluate_model(knn_model, X_test, y_test)
results['kNN'] = knn_results_test[:-1]
confusion_matrices['kNN'] = knn_results_test[-1]

# Print confusion matrices separately for clarity
for model_name, conf_matrix in confusion_matrices.items():
    print(f"\nConfusion Matrix for {model_name} (Test set):\n{conf_matrix}")

# Create a DataFrame to display the results
metrics_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T
print("Test Set Results:\n", metrics_df)

# Visualizing the training and test set results for each model
for model_name, model in {
    'Logistic Regression': log_reg_model,
    'Decision Tree': decision_tree_model,
    'Random Forest': random_forest_model,
    'SVM': svm_model,
    'kNN': knn_model
}.items():
    plt.figure(figsize=(12, 5))

    # Plot training set
    plt.subplot(1, 2, 1)
    x_set_train, y_set_train = X_train, y_train
    x1_train, x2_train = np.meshgrid(np.arange(start=x_set_train[:, 0].min() - 1, stop=x_set_train[:, 0].max() + 1, step=0.01),
                                     np.arange(start=x_set_train[:, 1].min() - 1, stop=x_set_train[:, 1].max() + 1, step=0.01))
    plt.contourf(x1_train, x2_train, model.predict(np.array([x1_train.ravel(), x2_train.ravel()]).T).reshape(x1_train.shape),
                 alpha=0.75, cmap=ListedColormap(('purple', 'green')))
    plt.xlim(x1_train.min(), x1_train.max())
    plt.ylim(x2_train.min(), x2_train.max())
    for i, j in enumerate(np.unique(y_set_train)):
        plt.scatter(x_set_train[y_set_train == j, 0], x_set_train[y_set_train == j, 1],
                    c=ListedColormap(('purple', 'green'))(i), label=j)
    plt.title(f'{model_name} (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()

    # Plot test set
    plt.subplot(1, 2, 2)
    x_set_test, y_set_test = X_test, y_test
    x1_test, x2_test = np.meshgrid(np.arange(start=x_set_test[:, 0].min() - 1, stop=x_set_test[:, 0].max() + 1, step=0.01),
                                   np.arange(start=x_set_test[:, 1].min() - 1, stop=x_set_test[:, 1].max() + 1, step=0.01))
    plt.contourf(x1_test, x2_test, model.predict(np.array([x1_test.ravel(), x2_test.ravel()]).T).reshape(x1_test.shape),
                 alpha=0.75, cmap=ListedColormap(('purple', 'green')))
    plt.xlim(x1_test.min(), x1_test.max())
    plt.ylim(x2_test.min(), x2_test.max())
    for i, j in enumerate(np.unique(y_set_test)):
        plt.scatter(x_set_test[y_set_test == j, 0], x_set_test[y_set_test == j, 1],
                    c=ListedColormap(('purple', 'green'))(i), label=j)
    plt.title(f'{model_name} (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()

    plt.show()

