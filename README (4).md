# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 13/05/2025
### Name: ASHOK KUMAR PREETHAM KUMAR
### Reg No: 212224040032

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

###  Import the neccessary packages
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
```

###  Load the dataset
```
data = pd.read_csv('AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])
```

### Label x and y axis 
```
plt.plot(data['Month'], data['#Passengers'])
plt.xlabel('Date')
plt.ylabel('No of Passengers')
plt.title('SARIMA model')
plt.show()
```
```
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['#Passengers'])
```
### Plot
```
plot_acf(data['#Passengers'])
plt.show()
plot_pacf(data['#Passengers'])
plt.show()
```
```
sarima_model = SARIMAX(data['#Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1,12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['#Passengers'][:train_size], data['#Passengers'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test)-1)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('No of Passengers')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```


### OUTPUT:

### SARIMA model

![image](https://github.com/user-attachments/assets/76495b66-8bb0-4b9d-a509-8dc3f5540567)

### ADF Statistic,p-value,Critical Values:

![image](https://github.com/user-attachments/assets/6c60fdeb-8dc7-4d32-850d-f1b41f8151a4)

### Autocorrelation

![image](https://github.com/user-attachments/assets/0510b63b-d441-49f9-bcc7-e28be51653b8)

### Partial Autocorrelation

![image](https://github.com/user-attachments/assets/65b101d0-6c40-4999-9cb0-08b501995dea)

### SARIMA Model Predictions

![image](https://github.com/user-attachments/assets/855c4721-a406-4b62-af7d-a337030183fd)



### RESULT:
Thus the program run successfully based on the SARIMA model.
