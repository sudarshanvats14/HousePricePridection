from keras.models import load_model
from data2 import X_test, y_test
import matplotlib.pyplot as plt



model = load_model("model.h5")  


y_val_pred = model.predict(X_test)


print("Predicted House Prices on Validation Set:")
print(y_val_pred[:5])


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_val_pred.flatten(), color='blue', alpha=0.5)
plt.title('Predicted vs Actual House Prices (Validation Set)')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.show()