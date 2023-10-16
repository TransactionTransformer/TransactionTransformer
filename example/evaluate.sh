category='phish-hack'

CUDA_VISIBLE_DEVICES=0 python ../Trans2Former/evaluate/evaluate.py \
--user-dir ../Trans2Former/ \
--num-workers 32 \
--ddp-backend=legacy_ddp \
--dataset-name ethereum \
--dataset-source pkl \
--pretrain-task NCP \
--data-dir ../dataset/finetune/{$category} \
--task finetune \
--max-edges 1024 \
--seed 666 \
--criterion binary_loss \
--arch trans2former_base \
--performer \
--performer-feature-redraw-interval 100 \
--pre-layernorm \
--num-classes 1 \
--batch-size 4 \
--data-buffer-size 16 \
--encoder-layers 12 \
--encoder-embed-dim 768 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 32 \
--remove-head \
--max-epoch 1000 \
--pretrained-model-name  finetune_$category \
--load-pretrained-model-output-layer \
--encoding-method concat \
--split test \
--metric auc_acc_recall_precision_f1 \
--is-evaluate

# 2023-10-06 15:48:48 | INFO | __main__ | auc: 0.9641957747220905                                                                     
# 2023-10-06 15:48:48 | INFO | __main__ | acc: 0.9029126167297363
# 2023-10-06 15:48:48 | INFO | __main__ | recall: 0.9032257795333862
# 2023-10-06 15:48:48 | INFO | __main__ | precision: 0.8842105269432068
# 2023-10-06 15:48:48 | INFO | __main__ | f1: 0.8936170339584351

# 2023-10-08 11:09:59 | INFO | __main__ | auc: 0.9641957747220905                                                  
# 2023-10-08 11:09:59 | INFO | __main__ | acc: 0.9029126167297363
# 2023-10-08 11:09:59 | INFO | __main__ | recall: 0.4516128897666931
# 2023-10-08 11:09:59 | INFO | __main__ | precision: 0.9032257795333862
# 2023-10-08 11:09:59 | INFO | __main__ | f1: 0.602150559425354