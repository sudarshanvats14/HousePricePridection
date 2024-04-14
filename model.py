import numpy as np
import matplotlib.pyplot as plt
from data2 import X_train, y_train, X_test, y_test
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras.callbacks import EarlyStopping


model = Sequential()
model.add(Dense(300, input_dim=19, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(300, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(300, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(300, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, batch_size=40, 
                    validation_data=(X_test, y_test), callbacks=[early_stopping])


model.save('model.h5')

y_pred_test = model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f'Final Test Metrics:')
print(f'Mean Squared Error: {mse_test}')
print(f'Root Mean Squared Error: {rmse_test}')
print(f'Mean Absolute Error: {mae_test}')
print(f'R-squared: {r2_test}')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

