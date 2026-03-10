def merge_maker_data(
    sales_df,
    maker_df,
    sales_maker_col="maker_name",
    maker_maker_col="maker_name",
    how="left",
    suffixes=("", "_maker"),
    normalize_names=True,
    verbose=True,
):
    """
    Merge maker-level data into sales-level data by maker name.

    If maker_df has multiple rows with the same maker name, this function keeps
    ONLY the first occurrence (in maker_df order) and discards the rest.

    Also drops the 'info' column from maker_df before merging (if present).
    """
    import pandas as pd
    import re

    s = sales_df.copy()
    m = maker_df.copy()

    if sales_maker_col not in s.columns:
        raise ValueError(f"'{sales_maker_col}' not found in sales_df")
    if maker_maker_col not in m.columns:
        raise ValueError(f"'{maker_maker_col}' not found in maker_df")

    # Drop maker 'info' column if present
    m = m.drop(columns=["info"], errors="ignore")

    def _norm(x):
        if pd.isna(x):
            return pd.NA
        x = str(x).strip().casefold()
        x = re.sub(r"\s+", " ", x)
        return x

    if normalize_names:
        s["_maker_key"] = s[sales_maker_col].map(_norm)
        m["_maker_key"] = m[maker_maker_col].map(_norm)
        key = "_maker_key"
    else:
        key = maker_maker_col
        s["_maker_key"] = s[sales_maker_col]
        m["_maker_key"] = m[maker_maker_col]

    # Keep first maker row per key (preserve maker_df order)
    if verbose:
        dup_counts = m["_maker_key"].value_counts(dropna=False)
        dups = dup_counts[dup_counts > 1]
        if len(dups):
            print("WARNING: maker_df has duplicate maker names. Keeping ONLY the first row for each name.")
            print(dups.head(20))

    m_first = m.drop_duplicates(subset=["_maker_key"], keep="first")

    merged = s.merge(
        m_first,
        left_on="_maker_key",
        right_on="_maker_key",
        how=how,
        suffixes=suffixes,
    )

    if verbose:
        maker_cols_added = [c for c in m_first.columns if c not in {maker_maker_col, "_maker_key"}]
        if maker_cols_added:
            matched = merged[maker_cols_added].notna().any(axis=1).sum()
            print(f"Merged maker data: matched {matched}/{len(merged)} sales rows.")
            if matched < len(merged):
                unmatched = merged.loc[~merged[maker_cols_added].notna().any(axis=1), sales_maker_col].dropna()
                if len(unmatched):
                    print("Top unmatched maker_name values (sales_df):")
                    print(unmatched.value_counts().head(15))

    # Drop temp key
    merged = merged.drop(columns=["_maker_key"], errors="ignore")

    return merged