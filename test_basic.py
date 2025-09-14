
from recsys.hybrid import HybridRecommender
from pathlib import Path

def test_recommend_user_movies():
    rec = HybridRecommender(Path("data"))
    out = rec.recommend_for_user("movies", "u1", top_n=3)
    assert isinstance(out, list)
    assert len(out) <= 3
    if len(out) > 0:
        assert "item_id" in out[0] and "score" in out[0]
