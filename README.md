# F1 race pace prediction

Per-driver **LSTM models** that predict Formula 1 race lap times from practice and qualifying sessions.
During the second half of the 2024 season, a prediction was published after each qualifying session and checked
against the real race. Kept as an example of time-series deep learning outside geospatial data.

## Method

- **Data:** practice (FP1–FP3) and qualifying lap times 2022–2024 per driver, collected with
  [FastF1](https://github.com/theOehrly/Fast-F1). Race laps are the target.
- **Model:** one LSTM network per driver. Hyperparameters (number of layers, units, dropout, learning rate,
  sequence length) are tuned with `keras-tuner`, using Huber loss. SHAP is used to inspect feature importance.
- **Post-processing:** outlier laps are replaced, and predicted lap times are smoothed with a simple Kalman filter.
- **Evaluation:** after each race, prediction vs reality with MAE, RMSE, MAPE and R², plus plots for every driver.

| File | Purpose |
|---|---|
| [`Model_F1.ipynb`](Model_F1.ipynb) | training and tuning of the per-driver models |
| [`Prediction_code.ipynb`](Prediction_code.ipynb) | prediction template for a race weekend |
| `prediction_GRAND_PRIX_*_2024.ipynb` | executed predictions and post-race checks: USA, Mexico, Brazil, Las Vegas, Qatar, Abu Dhabi |

## Run

The notebooks run in Google Colab. Training data and saved models were kept on Google Drive and are not
included in the repository, so the executed notebooks serve as the record of results.

```bash
pip install -r requirements.txt
```

## Known limitations

- Each driver has a small dataset, so the models overfit easily. Race events (safety cars, penalties, strategy)
  are outside the model and are only described qualitatively in each weekend summary.
- The data-collection step was run locally and is not part of the repository.

**Status:** 2024 season, not maintained.

## License

[MIT](LICENSE)
