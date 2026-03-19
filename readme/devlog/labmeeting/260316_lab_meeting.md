## 한 일
prompt 확정
- 별로 DrOp이 없어서 _icl2로 확정
global 검색으로 보완 가능?
- round6에서 escape
- 

TDOO 
leaf에 도달했을 때 다음 state 선택 구체화



## Experiments


baseline: round6-RInTP=-1-PlTau=5.0-DC=True-RCF=0.5-NumI=10-MaxBS=10-S=round6_mrr_selector_accum_meanscore_global_method2_expandable_pool_freeze_terminal-FT=1000-PreFRS=branch-RPN=agent_executor_v1_icl2-RM=concat-RE=1-RCT=10-RCS=mixed-RGT=10-RSM=meanscore_global-RM=True-RMM=expandable_pool-REPFTB=True-RRrfK=60-RRC=leaf-REM=replace



## Writing


260318
- Average next-step change after the first reseat:
  - `nDCG(iter t+1) - nDCG(iter t) = -2.69`

  Baseline per-iteration mean:

- `30.97, 31.26, 31.52, 31.68, 31.66, 31.41, 31.35, 31.42, 31.38, 31.40`

Round6 per-iteration mean:

- `31.07, 31.50, 31.45, 29.24, 29.63, 32.66, 32.42, 32.30, 32.25, 32.09`