######################################
# This file contains the code for splitting the data into train, validation and test sets.
# The split is stratified by both country and type, to ensure that all sets have similar distributions of these important categorical variables.
# We also fit maker-level location fill rules using only the training data, and then apply those rules to all splits to fill in missing location information based on maker name.
######################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("price_adj_w_all_features.csv")

def stratified_train_val_test_split(
    df,
    country_col="country_iso1",
    type_col="type",
    test_size=0.20,
    val_size=0.20,# fraction of the remaining train to use for validation
    min_count=5,  # minimum rows required per stratum; smaller strata get grouped to OTHER
    random_state=42,
):
    d = df.copy()

    # Fill missing strat columns so stratify doesn't error
    d[country_col] = d[country_col].fillna("UNK")
    d[type_col] = d[type_col].fillna("UNK")

    # Joint stratum
    d["_stratum"] = d[country_col].astype(str) + "||" + d[type_col].astype(str)

    # Collapse rare strata -> OTHER
    counts = d["_stratum"].value_counts()
    rare = counts[counts < min_count].index
    d.loc[d["_stratum"].isin(rare), "_stratum"] = "OTHER"

    # 1) train vs test
    train_df, test_df = train_test_split(
        d,
        test_size=test_size,
        random_state=random_state,
        stratify=d["_stratum"],
    )

    # 2) train vs val (within train_df)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df["_stratum"],
    )

    # Drop helper column
    for x in (train_df, val_df, test_df):
        x.drop(columns=["_stratum"], inplace=True)

    return train_df, val_df, test_df


train_df, val_df, test_df = stratified_train_val_test_split(
   data,
    country_col="country_iso1",
    type_col="type",
    test_size=0.20,
    val_size=0.20,   # 20% of remaining -> if test=0.2, then val is 0.2*0.8=0.16 of total
    min_count=5,
    random_state=42,
)

################## Next we fit maker-level location fill rules using only the training data, and then apply those rules to all splits to fill in missing location information based on maker name. ##################
def fit_maker_location_rules(
    train_df,
    maker_col="maker_name",
    city_col="city_maker",
    country_col="country_iso1",
    admin1_col="admin1_name",
    admin2_col="admin2_name",
    method="shared_level",   # "shared_level" or "mode"
):
    """
    Fit maker-level location fill rules using TRAIN ONLY.
    Returns a dataframe keyed by maker_col with columns:
      _fill_city, _fill_country, _fill_admin1, _fill_admin2, _single_city
    Assumes norm_text exists.
    """
    t = train_df.copy()

    for c in [maker_col, city_col, country_col, admin1_col, admin2_col]:
        if c not in t.columns:
            t[c] = pd.NA

    for c in [city_col, country_col, admin1_col, admin2_col]:
        t[c] = t[c].replace(r"^\s*$", pd.NA, regex=True)

    def _mode_nonnull(s):
        s2 = s.dropna()
        return pd.NA if s2.empty else s2.value_counts().idxmax()

    def _shared_or_na(s):
        s2 = s.dropna()
        u = pd.unique(s2)
        return u[0] if len(u) == 1 else pd.NA

    method = str(method).strip().casefold()
    if method not in {"shared_level", "mode"}:
        raise ValueError("method must be 'shared_level' or 'mode'")

    if method == "mode":
        rules = (
            t[t[maker_col].notna()]
            .groupby(maker_col, sort=False)
            .agg(
                _fill_city=(city_col, _mode_nonnull),
                _fill_country=(country_col, _mode_nonnull),
                _fill_admin1=(admin1_col, _mode_nonnull),
                _fill_admin2=(admin2_col, _mode_nonnull),
            )
            .reset_index()
        )
        rules["_single_city"] = True  # in mode method we allow filling city directly
        return rules

    # shared_level
    t["_city_key"] = t[city_col].map(lambda x: norm_text(x) if pd.notna(x) else pd.NA)

    def _maker_rule(g):
        city_keys = pd.unique(g["_city_key"].dropna())

        if len(city_keys) == 1:
            key = city_keys[0]
            gg = g[g["_city_key"] == key]
            return pd.Series(
                {
                    "_fill_city": _mode_nonnull(gg[city_col]),
                    "_fill_country": _mode_nonnull(gg[country_col]),
                    "_fill_admin1": _mode_nonnull(gg[admin1_col]),
                    "_fill_admin2": _mode_nonnull(gg[admin2_col]),
                    "_single_city": True,
                }
            )

        fill_country = _shared_or_na(g[country_col])
        fill_admin1 = _shared_or_na(g[admin1_col]) if pd.notna(fill_country) else pd.NA
        fill_admin2 = _shared_or_na(g[admin2_col]) if pd.notna(fill_admin1) else pd.NA

        return pd.Series(
            {
                "_fill_city": pd.NA,
                "_fill_country": fill_country,
                "_fill_admin1": fill_admin1,
                "_fill_admin2": fill_admin2,
                "_single_city": False,
            }
        )

    rules = (
        t[t[maker_col].notna()]
        .groupby(maker_col, sort=False)
        .apply(_maker_rule)
        .reset_index()
    )
    return rules.drop(columns=["_city_key"], errors="ignore")


def apply_maker_location_rules(
    df,
    rules_df,
    maker_col="maker_name",
    city_col="city_maker",
    country_col="country_iso1",
    admin1_col="admin1_name",
    admin2_col="admin2_name",
    filled_col="location_filled",
    only_when_city_and_location_missing=True,
):
    """
    Apply pre-fit maker rules to any dataframe (train/val/test) without refitting.
    Only fills NaNs; does not overwrite existing values.
    Adds filled_col indicating if any location field was imputed.
    """
    out = df.copy()

    for c in [maker_col, city_col, country_col, admin1_col, admin2_col]:
        if c not in out.columns:
            out[c] = pd.NA
    for c in [city_col, country_col, admin1_col, admin2_col]:
        out[c] = out[c].replace(r"^\s*$", pd.NA, regex=True)

    before = out[[city_col, country_col, admin1_col, admin2_col]].copy()

    out = out.merge(rules_df, on=maker_col, how="left")

    loc_all_missing = out[[country_col, admin1_col, admin2_col]].isna().all(axis=1)
    city_missing = out[city_col].isna()

    if only_when_city_and_location_missing:
        eligible = city_missing & loc_all_missing & out[maker_col].notna()
    else:
        eligible = (city_missing | loc_all_missing) & out[maker_col].notna()

    out.loc[eligible, country_col] = out.loc[eligible, country_col].fillna(out.loc[eligible, "_fill_country"])
    out.loc[eligible, admin1_col]  = out.loc[eligible, admin1_col].fillna(out.loc[eligible, "_fill_admin1"])
    out.loc[eligible, admin2_col]  = out.loc[eligible, admin2_col].fillna(out.loc[eligible, "_fill_admin2"])

    city_fill_mask = eligible & out[city_col].isna() & out["_single_city"].fillna(False)
    out.loc[city_fill_mask, city_col] = out.loc[city_fill_mask, "_fill_city"]

    cols = [city_col, country_col, admin1_col, admin2_col]
    sentinel = "__NA__"

    before_arr = before.reindex(columns=cols).fillna(sentinel).to_numpy()
    after_arr  = out[cols].fillna(sentinel).to_numpy()

    out[filled_col] = (before_arr != after_arr).any(axis=1)

    out = out.drop(columns=["_fill_city", "_fill_country", "_fill_admin1", "_fill_admin2", "_single_city"], errors="ignore")
    return out

for df_ in (train_df, val_df, test_df):
    df_["country_iso1"] = df_["country_iso1"].replace("UNK", pd.NA)
    df_["type"] = df_["type"].replace("UNK", pd.NA)

# Fit rules on TRAIN only
rules = fit_maker_location_rules(train_df, maker_col="maker_name", method="mode")

# Apply to each split (we can apply the same rules since they were derived from the train set)
train_df = apply_maker_location_rules(train_df, rules, maker_col="maker_name", filled_col="location_filled")
val_df   = apply_maker_location_rules(val_df,   rules, maker_col="maker_name", filled_col="location_filled")
test_df  = apply_maker_location_rules(test_df,  rules, maker_col="maker_name", filled_col="location_filled")

#Save Data
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)