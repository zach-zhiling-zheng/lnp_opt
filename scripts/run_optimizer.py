import argparse
from src.optimization.lnp_optimizer import main as opt_main

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4, help="Batch size per iteration")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--methods", type=str, default="Random,BO,EDBO,LLM,R-LLM")
    p.add_argument("--xlsx", type=str, default="LNP.xlsx", help="Path to your LNP.xlsx")
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    opt_main(args.n, args.iterations, args.repeat, methods, args.xlsx)
