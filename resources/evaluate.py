from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def evaluate(X_test, y_test, model):
  y_true = y_test
  y_pred = model.predict(X_test) > 0.5
  cm = confusion_matrix(y_true, y_pred)

  tp = cm[0][0]
  tn = cm[1][1]
  fp = cm[1][0]
  fn = cm[0][1]

  con_matrix = {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)

  return con_matrix, accuracy, precision, recall