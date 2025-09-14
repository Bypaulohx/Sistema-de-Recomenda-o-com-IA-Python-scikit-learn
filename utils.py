
import pandas as pd
from pathlib import Path

def load_catalog(domain: str, data_dir: Path) -> pd.DataFrame:
    if domain == "movies":
        df = pd.read_csv(data_dir / "movies.csv")
        df = df.rename(columns={"id":"item_id"})
        df["text"] = df["title"].fillna("") + " " + df["genres"].fillna("")
    elif domain == "music":
        df = pd.read_csv(data_dir / "music.csv")
        df = df.rename(columns={"id":"item_id"})
        df["text"] = df["title"].fillna("") + " " + df["artist"].fillna("") + " " + df["genres"].fillna("")
    elif domain == "products":
        df = pd.read_csv(data_dir / "products.csv")
        df = df.rename(columns={"id":"item_id","name":"title"})
        df["text"] = df["title"].fillna("") + " " + df["category"].fillna("") + " " + df["tags"].fillna("")
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return df

def load_interactions(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "interactions.csv")
