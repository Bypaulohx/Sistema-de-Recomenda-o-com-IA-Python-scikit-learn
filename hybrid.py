
from typing import List, Tuple, Dict, Optional
import pandas as pd
from pathlib import Path
from .utils import load_catalog, load_interactions
from .content import ContentRecommender
from .cf import CFRecommender

class HybridRecommender:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._catalog = {}
        self._content = {}
        self._cf = {}
        self._inter = load_interactions(self.data_dir)
        for domain in ["movies","music","products"]:
            items = load_catalog(domain, self.data_dir)
            self._catalog[domain] = items
            self._content[domain] = ContentRecommender(items)
            self._cf[domain] = CFRecommender(self._inter, items, domain)

    def _merge(self, a: List[Tuple[str, float]], b: List[Tuple[str, float]], wa: float, wb: float, top_n: int) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for iid, s in a:
            scores[iid] = scores.get(iid, 0.0) + wa * s
        for iid, s in b:
            scores[iid] = scores.get(iid, 0.0) + wb * s
        order = sorted(scores.items(), key=lambda x: -x[1])
        return order[:top_n]

    def recommend_for_user(self, domain: str, user_id: str, top_n: int = 10, weights: Optional[Dict[str, float]] = None):
        weights = weights or {"content": 0.5, "cf": 0.5}
        items = self._catalog[domain]
        # liked items: ratings >= 4
        liked = self._inter[(self._inter["domain"] == domain) & (self._inter["user_id"] == user_id) & (self._inter["rating"] >= 4)]
        liked_ids = liked["item_id"].tolist()
        content_recs = self._content[domain].recommend_for_profile(liked_ids, top_n=top_n*3)
        cf_recs = self._cf[domain].recommend_for_user(user_id, top_n=top_n*3)
        merged = self._merge(content_recs, cf_recs, weights.get("content",0.5), weights.get("cf",0.5), top_n)
        return self._attach_titles(items, merged)

    def recommend_similar(self, domain: str, item_id: str, top_n: int = 10):
        items = self._catalog[domain]
        content_recs = self._content[domain].recommend_similar(item_id, top_n=top_n)
        return self._attach_titles(items, content_recs)

    def _attach_titles(self, items: pd.DataFrame, recs: List[Tuple[str, float]]):
        meta = items.set_index("item_id")
        out = []
        for iid, score in recs:
            row = meta.loc[iid]
            title = row.get("title", row.get("name", str(iid)))
            out.append({"item_id": iid, "title": str(title), "score": float(score)})
        return out
