
import argparse, json
from recsys.hybrid import HybridRecommender
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="AI Recommender CLI")
    parser.add_argument("--domain", choices=["movies","music","products"], default="movies")
    parser.add_argument("--user", help="user id for user-based recommendations")
    parser.add_argument("--item", help="item id for similar-items recommendations")
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    rec = HybridRecommender(Path(args.data_dir))
    if args.user:
        out = rec.recommend_for_user(args.domain, args.user, top_n=args.topn)
    elif args.item:
        out = rec.recommend_similar(args.domain, args.item, top_n=args.topn)
    else:
        raise SystemExit("Provide --user or --item")
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
