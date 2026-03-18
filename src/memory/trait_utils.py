# src/memory/trait_utils.py

def rows_bucket(n: int) -> str:
    if n < 1000:
        return "small"
    if n < 50000:
        return "medium"
    return "large"


def features_bucket(n: int) -> str:
    if n < 10:
        return "small"
    if n < 50:
        return "medium"
    return "large"


def balance_bucket(ratio: float) -> str:
    if ratio >= 0.8:
        return "balanced"
    if ratio >= 0.4:
        return "moderate"
    return "severe"
