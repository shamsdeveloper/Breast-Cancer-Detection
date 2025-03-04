import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import pickle
class BreastCancerDetection:
    def __init__(self):
        self.data = None
        self.df = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def load_data(self):
        cancer_dataset = load_breast_cancer()
        self.data = cancer_dataset
        self.df = pd.DataFrame(
            np.c_[cancer_dataset['data'], cancer_dataset['target']],
            columns=np.append(cancer_dataset['feature_names'], ['target'])
        )
        print("Data loaded successfully.")

    def explore_data(self):
        print("Dataset Description:\n", self.data['DESCR'])
        print("Feature Names:\n", self.data['feature_names'])
        print("Target Names:\n", self.data['target_names'])

    def visualize_data(self):
        sns.pairplot(self.df, hue='target')
        plt.show()
        plt.figure(figsize=(16, 9))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', linewidths=2)
        plt.show()

    def preprocess_data(self):
        X = self.df.drop(['target'], axis=1)
        y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        self.X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_test_sc = self.scaler.transform(self.X_test)
        print("Data preprocessing completed.")

    def train_model(self, model_name, model, scaled=False):
        if scaled:
            model.fit(self.X_train_sc, self.y_train)
            y_pred = model.predict(self.X_test_sc)
        else:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        self.models[model_name] = model
        self.results[model_name] = accuracy
        print(f"{model_name} trained. Accuracy: {accuracy:.2f}")

    def tune_xgboost(self, param_grid, search_type='randomized'):
        xgb = XGBClassifier()
        if search_type == 'randomized':
            search = RandomizedSearchCV(xgb, param_distributions=param_grid, scoring='roc_auc', n_jobs=-1, verbose=3)
        elif search_type == 'grid':
            search = GridSearchCV(xgb, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, verbose=3)
        else:
            raise ValueError("search_type must be 'randomized' or 'grid'")

        search.fit(self.X_train, self.y_train)
        best_model = search.best_estimator_
        self.models['XGBoost_Tuned'] = best_model
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.results['XGBoost_Tuned'] = accuracy
        print("XGBoost tuned. Best Params:", search.best_params_)
        print(f"XGBoost Tuned Accuracy: {accuracy:.2f}")

    def evaluate_model(self, model_name):
        model = self.models.get(model_name)
        if model is None:
            print(f"Model {model_name} not found.")
            return

        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Confusion Matrix for {model_name}:\n", cm)
        print(f"Classification Report for {model_name}:\n", classification_report(self.y_test, y_pred))

    def save_model(self, model_name, file_name):
        model = self.models.get(model_name)
        if model:
            with open(file_name, 'wb') as file:
                pickle.dump(model, file)
            print(f"Model {model_name} saved as {file_name}.")
        else:
            print(f"Model {model_name} not found.")

    def load_model(self, file_name):
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_name}.")
        return model

if __name__ == "__main__":
    detector = BreastCancerDetection()
    detector.load_data()
    detector.explore_data()
    detector.preprocess_data()

    # Train various models
    detector.train_model("SVM", SVC(), scaled=True)
    detector.train_model("Logistic Regression", LogisticRegression(penalty='l1', solver='liblinear'), scaled=True)
    detector.train_model("KNN", KNeighborsClassifier(n_neighbors=5), scaled=True)
    detector.train_model("Naive Bayes", GaussianNB(), scaled=False)
    detector.train_model("Decision Tree", DecisionTreeClassifier(criterion='entropy'), scaled=False)
    detector.train_model("Random Forest", RandomForestClassifier(n_estimators=20, criterion='entropy'), scaled=False)
    detector.train_model("AdaBoost", AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy'), n_estimators=2000, learning_rate=0.1), scaled=False)
    detector.train_model("XGBoost", XGBClassifier(), scaled=False)

    # Hyperparameter tuning for XGBoost
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.15],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
        "colsample_bytree": [0.3, 0.4, 0.5]
    }
    detector.tune_xgboost(param_grid, search_type='randomized')

    # Evaluate models
    detector.evaluate_model("SVM")
    detector.evaluate_model("XGBoost_Tuned")

    # Save the best model
    detector.save_model("XGBoost_Tuned", "breast_cancer_detector.pickle")
