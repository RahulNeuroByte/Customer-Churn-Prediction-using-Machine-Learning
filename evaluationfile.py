from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_models(models,X_test, y_test):
    results = {}

    for name, model in models.items():
        print(f"Evaluating: {name}")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precison: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")


        results[name] = {
            "accuracy": acc,
            "precison": prec,
            "recall": rec,
            "f1_score": f1
        }

    return results