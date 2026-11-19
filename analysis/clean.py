"""Step 1. Load the raw results and build one clean table per objective.

Objective 1: plain and _tt expectimax play identical games, so one row per
(solver, board, seed) is kept; expectimax_d8 on 4x4 (25 games) is dropped.
Run every analysis script from inside this folder, in the order 1-7.
"""
import pandas as pd, re
import os
# Run from inside the analysis/ folder. The Objective 2 file is the clean copy (see data/readme.md).
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
SRC1 = os.path.join(DATA, 'objective1_results.csv.gz')
SRC2 = os.path.join(DATA, 'objective2_results.csv.gz')
D = 'data/'; os.makedirs(D, exist_ok=True)
def norm(s):
    return re.sub(r'_tt$','',s)
o1=pd.read_csv(SRC1); o2=pd.read_csv(SRC2)
o1['solver_raw']=o1.solver; o2['solver_raw']=o2.solver
o1['solver']=o1.solver.map(norm); o2['solver']=o2.solver.map(norm)
# Obj1: _tt and plain replay identical games per seed -> keep one row per (solver,board,seed), prefer plain
o1=o1.sort_values('solver_raw').drop_duplicates(['solver','board','seed'],keep='first')
o1=o1[~((o1.solver=='expectimax_d8')&(o1.board=='4x4'))]  # n=25, too small
assert not o2.duplicated(['solver','board','seed']).any()
for d,o in [(o1,1),(o2,2)]:
    d['objective']=o
    d['cells']=d.board.map(lambda b: int(b[0])*int(b[2]))
    m=d.solver.str.extract(r'^(expectimax|minimax|mcts|greedy|random)')[0]
    d['family']=m
    d['depth']=d.solver.str.extract(r'_d(\d+)')[0].astype(float)
    d['rollouts']=d.solver.str.extract(r'mcts_(\d+)')[0].astype(float)
    d['sec_per_move']=d.seconds/d.moves.clip(lower=1)
o1.to_pickle(D+"obj1.pkl"); o2.to_pickle(D+"obj2.pkl")
for d in (o1,o2):
    print(d.pivot_table(index='solver',columns='board',values='score',aggfunc='count').fillna(0).astype(int).to_string())
    print(d.groupby('board').moves.max().to_dict())
