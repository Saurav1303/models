from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print the results.
        """
        y_pred = self.model.predict(self.X_test)

        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Precision:", precision_score(self.y_test, y_pred))
        print("Recall:", recall_score(self.y_test, y_pred))
        print("F1-Score:", f1_score(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def save_metrics_to_csv(self, filename):
        """
        Save the evaluation metrics to a CSV file.
        """
        metrics = {
            'Accuracy': accuracy_score(self.y_test, self.model.predict(self.X_test)),
            'Precision': precision_score(self.y_test, self.model.predict(self.X_test)),
            'Recall': recall_score(self.y_test, self.model.predict(self.X_test)),
            'F1-Score': f1_score(self.y_test, self.model.predict(self.X_test))
        }

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(filename, index=False)

# Example usage:
# Assuming clf is your trained RandomForestClassifier and X_test, y_test are your test data and target variable
# evaluator = ModelEvaluator(clf, X_test, y_test)
# evaluator.evaluate_model()
# evaluator.save_metrics_to_csv('model_evaluation_metrics.csv')