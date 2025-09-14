
# Guia da API (FastAPI)

## Endpoints
- `GET /health` – status.
- `POST /recommend` – recomenda por usuário **ou** item similar.
- `POST /feedback` – registra nova interação e recarrega o modelo.

### Modelos de payload
```json
// /recommend
{
  "domain": "movies | music | products",
  "user_id": "opcional",
  "item_id": "opcional",
  "top_n": 10,
  "weights": {"content": 0.5, "cf": 0.5}
}
```

```json
// /feedback
{
  "user_id": "u123",
  "domain": "movies",
  "item_id": "m10",
  "rating": 5
}
```
