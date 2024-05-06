mpirun -n 8 python tools/train_new.py \
--batch_size 16 \
--batch_sum 256 \
--trainable_layers 3 \
--selector_bias 4.0 \
--norm_layer frozen_bn \
--max_epochs 120 \
--drop_epochs 90 \
--eval_step 1 \
--dataset_type refcocog \
--splitBy umd \
--translate \
--multi_scale \
--arch_loss_coef 0.1 \
--lr_base 1e-4 \
--lr_visual 1e-5 \
--lr_lang 1e-5 \
--comments "" \
--pretrained_path ""