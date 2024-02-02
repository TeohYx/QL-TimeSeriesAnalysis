import pandas as pd
import numpy as np






test_predictions = []
first_eval_batch = X_all[-1]
current_batch = first_eval_batch.reshape((1, 1000, 4))

for i in range(100):
  current_pred = model.predict(current_batch)[0]
  test_predictions.append(current_pred[0])
  print(current_pred[0])
  current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)