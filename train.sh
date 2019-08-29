#!/usr/bin/env bash

python trainRotation.py --model_save_path=fourRotations
python trainClasiffiers.py \
--model_save_path=fc3_stop_gradient_4rotations \
--model_restore_path=fourRotations \
--fc_layers_number=3 \
--freeze_weights='True'