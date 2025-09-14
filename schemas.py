
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class RecommendRequest(BaseModel):
    domain: str = Field(..., description="one of: movies, music, products")
    user_id: Optional[str] = None
    item_id: Optional[str] = None
    top_n: int = 10
    weights: Dict[str, float] = {"content": 0.5, "cf": 0.5}

class Feedback(BaseModel):
    user_id: str
    domain: str
    item_id: str
    rating: float
