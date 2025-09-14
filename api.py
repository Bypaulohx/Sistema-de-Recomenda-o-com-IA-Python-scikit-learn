
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .hybrid import HybridRecommender
from .schemas import RecommendRequest, Feedback
from pathlib import Path
import pandas as pd

def create_app(data_dir: str = "data") -> FastAPI:
    app = FastAPI(title="AI Recommender (scikit-learn)", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    rec = HybridRecommender(Path(data_dir))

    @app.get("/health")
    def health():
        return {"status":"ok"}

    @app.post("/recommend")
    def recommend(req: RecommendRequest):
        domain = req.domain
        if req.user_id:
            out = rec.recommend_for_user(domain, req.user_id, top_n=req.top_n, weights=req.weights)
            return {"domain": domain, "user_id": req.user_id, "items": out}
        if req.item_id:
            out = rec.recommend_similar(domain, req.item_id, top_n=req.top_n)
            return {"domain": domain, "item_id": req.item_id, "items": out}
        raise HTTPException(status_code=400, detail="Provide user_id or item_id")

    @app.post("/feedback")
    def feedback(f: Feedback):
        # append to CSV and re-init
        df_new = pd.DataFrame([f.dict()])
        path = Path(data_dir) / "interactions.csv"
        df_old = pd.read_csv(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(path, index=False)
        nonlocal rec
        rec = HybridRecommender(Path(data_dir))
        return {"status":"recorded"}

    return app
