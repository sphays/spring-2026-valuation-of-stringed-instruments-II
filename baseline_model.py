import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def fit_baseline(train, min_count=5):
    """
    Compute mean prices on the training set.
    Returns three objects: maker_type_mean, type_mean, global_mean.
    min_count : minimum number of sales required to use the maker+type mean.
    """
    maker_type_stats = (
        train.groupby(['maker_id', 'type'])['price_usd_real']
        .agg(['mean', 'count'])
    )
    maker_type_mean = (
        maker_type_stats[maker_type_stats['count'] >= min_count]['mean']
        .to_dict()
    )

    # Fallback: mean by instrument type
    type_mean = train.groupby('type')['price_usd_real'].mean().to_dict()

    # Global fallback
    global_mean = train['price_usd_real'].mean()

    return maker_type_mean, type_mean, global_mean


def predict_baseline(df, maker_type_mean, type_mean, global_mean):
    """
    For each row:
      1. Use the (maker_id, type) mean if >= min_count sales in train
      2. Otherwise use the type mean
      3. Otherwise use the global mean
    """
    preds = []
    for _, row in df.iterrows():
        key = (row['maker_id'], row['type'])
        if key in maker_type_mean:
            preds.append(maker_type_mean[key])
        elif row['type'] in type_mean:
            preds.append(type_mean[row['type']])
        else:
            preds.append(global_mean)
    return np.array(preds)


def evaluate_baseline(train, val, target='price_usd_real', log=True, min_count=5):
    """
    Fit the baseline on train, evaluate on val.
    min_count : minimum number of sales to use the maker+type mean.
    If log=True, also reports metrics on the log scale.
    """
    maker_type_mean, type_mean, global_mean = fit_baseline(train, min_count)

    results = {}
    for name, df in [('val', val)]:
        y_true = df[target].values
        y_pred = predict_baseline(df, maker_type_mean, type_mean, global_mean)

        if log:
            log_true = np.log1p(y_true)
            log_pred = np.log1p(y_pred)
            rmse_log = mean_squared_error(log_true, log_pred) ** 0.5
            r2_log   = r2_score(log_true, log_pred)
        else:
            rmse_log = r2_log = None

        rmse  = mean_squared_error(y_true, y_pred) ** 0.5
        r2    = r2_score(y_true, y_pred)
        mape  = np.median(np.abs(y_pred - y_true) / y_true) * 100

        results[name] = {
            'RMSE':     rmse,
            'R2':       r2,
            'MedAPE':   mape,
            'RMSE_log': rmse_log,
            'R2_log':   r2_log,
        }

        print(f"\n── {name} ──")
        print(f"  R²       = {r2:.3f}")
        print(f"  RMSE     = {rmse:,.0f} USD")
        print(f"  MedAPE   = {mape:.1f}%")
        if log:
            print(f"  R²(log)  = {r2_log:.3f}")
            print(f"  RMSE(log)= {rmse_log:.3f}")

        n_maker_type = sum(
            1 for _, row in df.iterrows()
            if (row['maker_id'], row['type']) in maker_type_mean
        )
        n_type = len(df) - n_maker_type
        print(f"  Fallback: {n_maker_type} maker+type  |  {n_type} type only")
        print(f"  (min_count={min_count})")

    return results


# ── Usage ─────────────────────────────────────────────────────────────────────
# results = evaluate_baseline(train_all, val_all)
# results = evaluate_baseline(train_all, val_all, min_count=3)