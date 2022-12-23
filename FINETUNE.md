### Supervised finetuning ViT

For ViT-Base on 1% ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_finetune.py \
  --model vit_base_patch16 \
  --trainindex train_1p_index.csv \
  --finetune ./pretrain_weights/mae_pretrain_vit_base.pth \
  --batch_size 64 \
  --epochs 100 \
  --blr 0.00005 --layer_decay 0.65 \
  --weight_decay 0.05 --drop_path 0 --reprob 0 --mixup 0 --cutmix 0 \
  --dist_eval --eval_freq 5
```

For ViT-Base on 10% ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_finetune.py \
  --model vit_base_patch16 \
  --trainindex train_10p_index.csv \
  --finetune ./pretrain_weights/mae_pretrain_vit_base.pth \
  --batch_size 64 \
  --epochs 100 \
  --blr 0.00025 --layer_decay 0.65 \
  --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --dist_eval --eval_freq 5
```

For ViT-Large on 1% ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_finetune.py \
  --model vit_large_patch16 \
  --trainindex train_1p_index.csv \
  --finetune ./pretrain_weights/mae_pretrain_vit_large.pth \
  --batch_size 64 \
  --epochs 50 \
  --blr 0.001 --layer_decay 0.75 \
  --weight_decay 0.05 --drop_path 0 --reprob 0 --mixup 0 --cutmix 0 \
  --dist_eval --eval_freq 5
```

For ViT-Large on 10% ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_finetune.py \
  --model vit_large_patch16 \
  --trainindex train_10p_index.csv \
  --finetune ./pretrain_weights/mae_pretrain_vit_large.pth \
  --batch_size 64 \
  --epochs 50 \
  --blr 0.001 --layer_decay 0.75 \
  --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --dist_eval --eval_freq 5
```

For ViT-Huge on 1% ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_finetune.py \
  --model vit_huge_patch14 \
  --trainindex train_1p_index.csv \
  --finetune ./pretrain_weights/mae_pretrain_vit_huge.pth \
  --batch_size 16 \
  --epochs 50 \
  --cls_token \
  --blr 0.01 --layer_decay 0.75 \
  --weight_decay 0.05 --drop_path 0 --reprob 0 --mixup 0 --cutmix 0 \
  --dist_eval --eval_freq 5
```

For ViT-Huge on 10% ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_finetune.py \
  --model vit_huge_patch14 \
  --trainindex train_10p_index.csv \
  --finetune ./pretrain_weights/mae_pretrain_vit_huge.pth \
  --batch_size 16 \
  --epochs 50 \
  --cls_token \
  --blr 0.001 --layer_decay 0.75 \
  --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --model_ema --model_ema_decay 0.9998 \
  --dist_eval --eval_freq 5
```

### Supervised training ViT from scratch

For ViT-Base on 10% ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_finetune.py \
  --model vit_base_patch16 \
  --trainindex train_10p_index.csv \
  --batch_size 128 \
  --epochs 500 --warmup_epochs 50 \
  --blr 0.0001 --layer_decay 1.0 \
  --opt_betas 0.9 0.95 \
  --weight_decay 0.3 \
  --use_fixed_pos_emb \
  --model_ema \
  --dist_eval --eval_freq 5 --save_ckpt_freq 250
```

### Supervised training ConvNeXT from scratch

For ConvNeXT-Tiny on 10% ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
  ./main_conv.py \
  --model convnext_tiny --drop_path 0.1 \
  --trainindex train_10p_index.csv \
  --epochs 500 --warmup_epochs 50 \
  --batch_size 128 --lr 1e-3 --update_freq 1 \
  --use_amp true \
  --model_ema true --model_ema_eval true \
  --eval_freq 5 --save_ckpt_freq 250
```
