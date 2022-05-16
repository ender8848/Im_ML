# Part2 README

## Using the given hyperparameters to train a specific model

All hyperparameters have default value except x, so the simplest version of initialize the model is as follow:
```
regressor = Regressor(x_train)
regressor.fit(x_train, y_train)
```

However if you want to set some specific parameters, the model can be initialized as below:
```
regressor = Regressor(
                 x_train,
                 nb_epoch=800,
                 batch_size=800,
                 lr=0.01,
                 textual_value='ocean_proximity',
                 neurons=[20,20,12,6,16],
                 activations=["relu", "relu", "relu","relu","relu"],
                 shuffle=True
                 )
regressor.fit(x_train, y_train)
```

## Evaluate

MSE, MAE, MAPE can be calculated using the function regressor.score. We determine to use MSE as the return value to measure the quality of the model. The score can be written as follow:

```
error = regressor.score(x_test, y_test)
```

Tuning hyperparameters
instead of setting hyperparameters manually, we can achieve this goal using the function RegressorHyperParameterSearch(). The logic has been described in the report. This function can be called seperately.

