### Semi-supervised finetuning ViT

For ViT-Base on 1% ImageNet, assuming the supervised finetuned model in <path_to_finetune_base_1p>, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_semi.py \
  --model vit_base_patch16 \
  --super_finetune <path_to_finetune_base_1p>/checkpoint-99.pth \
  --trainindex_x train_1p_index.csv --trainindex_u train_99p_index.csv \
  --batch_size 16 \
  --epochs 100 \
  --model_ema --ema_teacher \
  --threshold 0.6 --lambda_u 5 \
  --drop_path 0 --reprob 0 \
  --disable_x_mixup --pseudo_mixup --pseudo_mixup_func ProbPseudoMixup \
  --dist_eval
```

For ViT-Base on 10% ImageNet, assuming the supervised finetuned model in <path_to_finetune_base_10p>, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_semi.py \
  --model vit_base_patch16 \
  --trainindex_x train_10p_index.csv --trainindex_u train_90p_index.csv \
  --super_finetune <path_to_finetune_base_10p>/checkpoint-99.pth \
  --batch_size 16 \
  --epochs 100 \
  --model_ema --ema_teacher \
  --threshold 0.5 --lambda_u 5 \
  --pseudo_mixup --pseudo_mixup_func ProbPseudoMixup \
  --dist_eval
```

For ViT-Large on 1% ImageNet, assuming the supervised finetuned model in <path_to_finetune_large_1p>, run:
```
python -m torch.distributed.launch --nnodes 2 --node_rank 0 \
  --nproc_per_node=8 --use_env --master_addr xx.xx.xx.xx \
  ./main_semi.py \
  --model vit_large_patch16 \
  --trainindex_x train_1p_index.csv --trainindex_u train_99p_index.csv \
  --super_finetune <path_to_finetune_large_1p>/checkpoint-49.pth \
  --batch_size 8 \
  --epochs 100 \
  --model_ema --ema_teacher \
  --threshold 0.6 --lambda_u 5 \
  --drop_path 0.1 \
  --pseudo_mixup --pseudo_mixup_func ProbPseudoMixup \
  --dist_eval
```
We use 2 machines to run ViT-Large experiments, and this script is to run on the first machine. Change ``--master_addr`` to the IP address of the first machine, and ``--node_rank`` to 1, and run that script on the second machine.

For ViT-Large on 10% ImageNet, assuming the supervised finetuned model in <path_to_finetune_large_10p>, run:
```
python -m torch.distributed.launch --nnodes 2 --node_rank 0 \
  --nproc_per_node=8 --use_env --master_addr xx.xx.xx.xx \
  ./main_semi.py \
  --model vit_large_patch16 \
  --trainindex_x train_10p_index.csv --trainindex_u train_90p_index.csv \
  --super_finetune <path_to_finetune_large_10p>/checkpoint-49.pth \
  --batch_size 8 \
  --epochs 100 \
  --blr 0.002 \
  --model_ema --ema_teacher \
  --threshold 0.6 --lambda_u 5 \
  --drop_path 0.2 \
  --pseudo_mixup --pseudo_mixup_func ProbPseudoMixup \
  --dist_eval
```
We use 2 machines to run ViT-Large experiments, and this script is to run on the first machine. Change ``--master_addr`` to the IP address of the first machine, and ``--node_rank`` to 1, and run that script on the second machine.

For ViT-Huge on 1% ImageNet, assuming the supervised finetuned model in <path_to_finetune_huge_1p>, run:
```
python -m torch.distributed.launch --nnodes 4 --node_rank 0 \
  --nproc_per_node=8 --use_env --master_addr xx.xx.xx.xx \
  ./main_semi.py \
  --model vit_huge_patch14 \
  --trainindex_x train_1p_index.csv --trainindex_u train_99p_index.csv \
  --super_finetune <path_to_finetune_huge_1p>/checkpoint-49.pth \
  --cls_token \
  --batch_size 2 \
  --epochs 50 \
  --blr 0.005 \
  --model_ema --ema_teacher \
  --threshold 0.7 --lambda_u 5 \
  --drop_path 0.05 \
  --pseudo_mixup --pseudo_mixup_func ProbPseudoMixup \
  --dist_eval
```
We use 4 machines to run ViT-Huge experiments, and this script is to run on the first machine. Change ``--master_addr`` to the IP address of the first machine, and ``--node_rank`` to 1/2/3, and run that script on the other machines.

For ViT-Huge on 10% ImageNet, assuming the supervised finetuned model in <path_to_finetune_huge_10p>, run:
```
python -m torch.distributed.launch --nnodes 4 --node_rank 0 \
  --nproc_per_node=8 --use_env --master_addr xx.xx.xx.xx \
  ./main_semi.py \
  --model vit_huge_patch14 \
  --trainindex_x train_10p_index.csv --trainindex_u train_90p_index.csv \
  --super_finetune <path_to_finetune_huge_10p>/checkpoint-49.pth \
  --cls_token \
  --batch_size 2 --accum_iter 2 \
  --epochs 50 \
  --blr 0.0025 \
  --model_ema --ema_teacher \
  --threshold 0.6 --lambda_u 5 \
  --pseudo_mixup --pseudo_mixup_func ProbPseudoMixup \
  --dist_eval
```
We use 4 machines to run ViT-Huge experiments, and this script is to run on the first machine. Change ``--master_addr`` to the IP address of the first machine, and ``--node_rank`` to 1/2/3, and run that script on the other machines.

### Semi-supervised finetuning ViT (from scratch)

For ViT-Base on 10% ImageNet, assuming the trained from scratch model in <path_to_scratch_base_10p>, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_semi.py \
  --model vit_base_patch16 \
  --trainindex_x train_10p_index.csv --trainindex_u train_90p_index.csv \
  --batch_size 16 \
  --epochs 100 \
  --model_ema --ema_teacher \
  --threshold 0.5 --lambda_u 5 \
  --pseudo_mixup --pseudo_mixup_func ProbPseudoMixup \
  --use_fixed_pos_emb \
  --layer_decay 0.85 \
  --super_finetune <path_to_scratch_base_10p>/checkpoint-499.pth \
  --dist_eval
```

### Semi-supervised finetuning ConvNeXT (from scratch)

For ConvNeXT-Tiny on 10% ImageNet, assuming the trained from scratch model in <path_to_scratch_convnext_tiny_10p>, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_semi_conv.py \
  --model convnext_tiny \
  --trainindex_x train_10p_index.csv --trainindex_u train_90p_index.csv \
  --batch_size 16 --lr 0.001 \
  --model_ema true --ema_teacher \
  --drop_path 0.1 \
  --pseudo_mixup --pseudo_mixup_func ProbPseudoMixup \
  --finetune <path_to_scratch_convnext_tiny_10p>/checkpoint.pth
```
