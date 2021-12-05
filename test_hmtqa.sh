CUDA_VISIBLE_DEVICES=1 python -m qa.table.experiments test --model=qa/runs/hmtqa/model.best.bin --test-file=data/processed_input/test_samples.jsonl --cuda --seed=0
