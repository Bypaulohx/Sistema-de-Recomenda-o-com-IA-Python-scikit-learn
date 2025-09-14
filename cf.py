
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class CFRecommender:
    def __init__(self, interactions: pd.DataFrame, items: pd.DataFrame, domain: str):
        # interactions: columns [user_id, domain, item_id, rating]
        df = interactions[interactions["domain"] == domain].copy()
        users = sorted(df["user_id"].unique().tolist())
        items_list = items["item_id"].tolist()
        item_index = {iid: i for i, iid in enumerate(items_list)}
        user_index = {u: i for i, u in enumerate(users)}

        rows, cols, vals = [], [], []
        for _, r in df.iterrows():
            if r["item_id"] not in item_index:
                continue
            rows.append(user_index[r["user_id"]])
            cols.append(item_index[r["item_id"]])
            vals.append(float(r["rating"]))
        if not rows:
            shape = (len(users), len(items_list))
            mat = csr_matrix(shape)
        else:
            mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items_list)))
        # store
        self.users = users
        self.items = items_list
        self.item_index = item_index
        self.user_index = user_index
        # Fit KNN on item vectors (transpose to item-user space)
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.item_user_matrix = mat.transpose().tocsr()
        self.model.fit(self.item_user_matrix)
        self.user_item_matrix = mat.tocsr()

    def recommend_for_user(self, user_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if user_id not in self.user_index:
            return []
        uidx = self.user_index[user_id]
        user_vec = self.user_item_matrix[uidx]
        # score items by similarity to items the user has rated (item-based CF)
        scores = np.zeros(len(self.items), dtype=float)
        rated_items = user_vec.indices
        rated_vals = user_vec.data
        for i, r in zip(rated_items, rated_vals):
            distances, neighbors = self.model.kneighbors(self.item_user_matrix[i], n_neighbors=min(20, len(self.items)))
            sims = 1.0 - distances.ravel()
            for j, s in zip(neighbors.ravel(), sims):
                if j in rated_items:
                    continue
                scores[j] += s * r
        # rank
        order = np.argsort(-scores)
        recs = []
        for j in order:
            if scores[j] <= 0:
                continue
            recs.append((self.items[j], float(scores[j])))
            if len(recs) >= top_n:
                break
        return recs
