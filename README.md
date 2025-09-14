# Sistema de Recomendação com IA (Python + scikit-learn)

Projeto completo com **recomendador híbrido** (Conteúdo + Colaborativo) para **filmes, músicas e produtos**. Inclui API com FastAPI, CLI, testes, Docker e exemplos de dados.

## ⚙️ Tecnologias
- Python 3.11
- scikit-learn (TF‑IDF + kNN com similaridade coseno)
- FastAPI + Uvicorn
- pandas / numpy / scipy
- PyTest
- Docker (opcional)

## Estrutura
```
recommender_ai/
├─ app.py
├─ recsys/
│  ├─ api.py
│  ├─ content.py
│  ├─ cf.py
│  ├─ hybrid.py
│  ├─ utils.py
│  └─ schemas.py
├─ data/
│  ├─ movies.csv
│  ├─ music.csv
│  ├─ products.csv
│  └─ interactions.csv
├─ tests/
│  └─ test_basic.py
├─ docs/
│  ├─ architecture.png
│  └─ demo_recs.png
├─ .vscode/
│  ├─ settings.json
│  └─ launch.json
├─ requirements.txt
├─ Dockerfile
└─ docker-compose.yml
```

## Arquitetura

- **Content-based**: TF‑IDF sobre textos (título + gêneros/tags) → similaridade coseno.
- **CF (Item-based)**: kNN em matriz item×usuário → soma ponderada por notas do usuário.
- **Híbrido**: mistura ponderada (default 50/50).

## 🚀 Rodando no VSCode (passo a passo)
1. **Clone** este repositório ou extraia o ZIP.
2. Abra a pasta `recommender_ai/` no **VSCode**.
3. Crie o ambiente:
   - **Windows (PowerShell)**:
     ```ps
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     python -m pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - **Linux/macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     python -m pip install --upgrade pip
     pip install -r requirements.txt
     ```
4. **Executar API**:
   - Use o atalho de debug **“Run API (Uvicorn)”** no VSCode **ou**:
     ```bash
     uvicorn recsys.api:create_app --factory --host 0.0.0.0 --port 8000 --reload
     ```
   - Teste no navegador: `http://localhost:8000/health`
   - Documentação: `http://localhost:8000/docs`
5. **CLI (exemplos)**:
   ```bash
   python app.py --domain movies --user u1 --topn 5
   python app.py --domain movies --item m1 --topn 5
   python app.py --domain music  --user u6 --topn 5
   python app.py --domain products --user u4 --topn 5
   ```
6. **Rodar testes**:
   ```bash
   pytest -q
   ```

## API – exemplos
### Recomendação por usuário
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"domain\":\"movies\", \"user_id\":\"u1\", \"top_n\":5}"
```
### Itens similares
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"domain\":\"products\", \"item_id\":\"p1\", \"top_n\":5}"
```
### Feedback online (aprendizado incremental simples)
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"u11\", \"domain\":\"movies\", \"item_id\":\"m3\", \"rating\":5}"
```

## Print (exemplo de recomendações)

## Como funciona
- **Content-based** usa `TfidfVectorizer(1,2)` no campo `text` de cada domínio.
- **CF item-based** usa `NearestNeighbors(metric="cosine")` sobre item×usuário.
- **Híbrido** une listas com pesos `{content: 0.5, cf: 0.5}` e ordena.

## Personalização
- Edite datasets em `data/*.csv`.
- Ajuste pesos no body da API: `"weights": {"content": 0.7, "cf": 0.3}`.
- Aumente `n_neighbors` no CF para catálogos maiores.

## Docker (opcional)
```bash
docker compose up --build
# API em http://localhost:8000
```
