from typing import Any, Dict, List


def builder_stats_to_dataframe(stats: List[Dict[str, Any]]) -> "pandas.DataFrame":  # noqa: F821
    """
    Processes the statistics produced by a builder.

    :param stats: list of observations
    :return: pandas.DataFrame
    """
    import pandas

    df = pandas.DataFrame(stats)
    for c in ["added", "removed"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    agg_columns = {
        k: v
        for k, v in {
            "added": "sum",
            "removed": "sum",
            "time_in": "sum",
            "iteration": "max",
        }.items()
        if k in df.columns
    }
    if not agg_columns or "pattern" not in df.columns:
        # nothing to do
        return df

    agg = df.groupby("pattern")[list(agg_columns.keys())].agg(agg_columns)
    for c in {"added", "removed", "time_in"} & set(agg_columns):
        agg = agg[agg[c] > 0]
    agg = agg.sort_values("time_in", ascending=False)
    return agg
