OMP_NUM_THREADS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DIVICES=0,1,2,3 python -m qa.table.experiments --train  --config_file="qa/config/config.vanilla_bert.json" --cuda
