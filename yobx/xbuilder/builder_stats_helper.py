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
        df[c] = df[c].fillna(0).astype(int)
    agg = df.groupby("pattern")[["added", "removed", "time_in", "iteration"]].agg(
        {"added": "sum", "removed": "sum", "time_in": "sum", "iteration": "max"}
    )
    agg = agg[(agg["added"] > 0) | (agg["removed"] > 0) | (agg["time_in"] > 0)].sort_values(
        "time_in", ascending=False
    )
    return agg
