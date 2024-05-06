mpirun -n 4 python /home/zekang/vglaw/src/ms_train.py \
--batch_size=1 \
--batch_sum=256 \
--dataset=refcocog \
--splitBy=umd \
--experiment_name=mindspore_vglaw_refcocog \
--short_comment=mindspore_vglaw_refcocog \
--law_type=svd \
--img_size=448 \
--vit_model=vitdet_b_mrcnn \
--pretrained_path='/home/zekang/vglaw/logdir/refcocog/epoch-latest_in_train_4.ckpt' \
--translate \
--lr_lang=1e-6 \
--lr_visual=1e-6 \
--lr_base=1e-5 \
--lr_scheduler=step \
--max_epochs=20 \
--drop_epochs=10 \
--log_freq=10 \
--use_mask \
--mode_name=PYNATIVE \
--save_ckpt_dir=/home/zekang/vglaw/logdir/refcocog