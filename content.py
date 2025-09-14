
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, items: pd.DataFrame):
        # items: columns [item_id, title, text]
        self.items = items.reset_index(drop=True).copy()
        self.item_index = {iid: idx for idx, iid in enumerate(self.items["item_id"].tolist())}
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.item_matrix = self.vectorizer.fit_transform(self.items["text"].fillna(""))

    def recommend_similar(self, item_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if item_id not in self.item_index:
            return []
        idx = self.item_index[item_id]
        sims = cosine_similarity(self.item_matrix[idx], self.item_matrix).ravel()
        order = np.argsort(-sims)
        recs = []
        for j in order:
            if j == idx:
                continue
            recs.append((self.items.loc[j, "item_id"], float(sims[j])))
            if len(recs) >= top_n:
                break
        return recs

    def recommend_for_profile(self, liked_item_ids: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        valid = [iid for iid in liked_item_ids if iid in self.item_index]
        if not valid:
            return []
        idxs = [self.item_index[iid] for iid in valid]
        profile = self.item_matrix[idxs].mean(axis=0)
        sims = cosine_similarity(profile, self.item_matrix).ravel()
        # exclude liked items
        exclude = set(valid)
        order = np.argsort(-sims)
        recs = []
        for j in order:
            iid = self.items.loc[j, "item_id"]
            if iid in exclude:
                continue
            recs.append((iid, float(sims[j])))
            if len(recs) >= top_n:
                break
        return recs
