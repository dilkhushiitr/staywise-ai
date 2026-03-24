"""
main.py — StayWise AI Entry Point

Usage:
    python3 main.py --step 1    # Data Pipeline
    python3 main.py --step 2    # RAG System
    python3 main.py --step 3    # LLM Integration
    python3 main.py --step 4    # Ranking Engine
    python3 main.py --step 5    # FastAPI Server
"""

import argparse, sys

def parse_args():
    parser = argparse.ArgumentParser(description="StayWise AI")
    parser.add_argument("--step", type=int, default=1, choices=[1,2,3,4,5],
                        help="1=Pipeline 2=RAG 3=LLM 4=Ranking 5=API")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.step == 1:
        from src.pipeline.runner import run_step1
        run_step1()
    elif args.step == 2:
        from src.rag.rag_runner import run_step2
        run_step2()
    elif args.step == 3:
        from src.llm.llm_runner import run_step3
        run_step3()
    elif args.step == 4:
        from src.ranking.ranking_runner import run_step4
        run_step4()
    elif args.step == 5:
        from src.api.api_runner import run_step5
        run_step5()
    else:
        print(f"Step {args.step} not yet implemented.")
        sys.exit(1)

if __name__ == "__main__":
    main()
