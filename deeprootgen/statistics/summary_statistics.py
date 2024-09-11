"""Contains utilities for producing summary statistics.

This module defines utility functions for producing summary statistics that can be used to compare simulated and observational data.
"""


def get_summary_statistics() -> list[dict]:
    """Get a list of available summary statistics and labels.

    Returns:
        list[dict]:
            A list of available summary statistics and labels.
    """
    summary_statistics: list[str] = [
        "average_rld",
        "total_rld",
        "average_root_length",
        "total_root_length",
    ]

    summary_statistic_list = []
    for summary_statistic in summary_statistics:
        label = summary_statistic.replace("_", " ").title()
        summary_statistic_list.append({"value": summary_statistic, "label": label})

    return summary_statistic_list
