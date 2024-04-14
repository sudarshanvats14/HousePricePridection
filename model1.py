import numpy as np
import matplotlib.pyplot as plt
from data import X_train, y_train, X_test, y_test
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.callbacks import EarlyStopping


X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

best_model = None
best_val_loss = float('inf')
all_val_losses = []


model = Sequential()
model.add(Dense(100, input_dim=7, activation='relu', kernel_regularizer=l2(0.1)))
model.add(Dense(30, activation='relu', kernel_regularizer=l2(0.1)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X[train_index], X[test_index]
    y_train_fold, y_val_fold = y[train_index], y[test_index]

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, 
                        validation_data=(X_val_fold, y_val_fold), 
                        callbacks=[early_stopping], verbose=0)

    best_epoch = np.argmin(history.history['val_loss'])
    val_loss = history.history['val_loss'][best_epoch]

    print(f'Fold {fold_index + 1} - Best Validation Loss: {val_loss}')

    all_val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

best_model.save("best_model.h5")

for fold_index, val_loss in enumerate(all_val_losses):
    plt.plot(history.history['loss'], label=f'Training Loss - Fold {fold_index + 1}')
    plt.plot(history.history['val_loss'], label=f'Validation Loss - Fold {fold_index + 1}')
    plt.title(f'Fold {fold_index + 1} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

y_pred_test = best_model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f'Final Test Metrics:')
print(f'Mean Squared Error: {mse_test}')
print(f'Root Mean Squared Error: {rmse_test}')
print(f'Mean Absolute Error: {mae_test}')
print(f'R-squared: {r2_test}')

plt.scatter(y_test, y_pred_test)
plt.title('Actual vs Predicted Values on Test Set')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
