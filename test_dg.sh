# ResNet50 backbone, ImageNet to its variants
# Before running the command, you need specify an evaluation output directory and the folder where the pretrianed mdoel is located
bash scripts/taskres/eval.sh imagenetv2 generalization_rn50 your_evaluation_output_path folder_ckpt_located
bash scripts/taskres/eval.sh imagenet_sketch generalization_rn50 your_evaluation_output_path folder_ckpt_located
bash scripts/taskres/eval.sh imagenet_a generalization_rn50 your_evaluation_output_path folder_ckpt_located
bash scripts/taskres/eval.sh imagenet_r generalization_50 your_evaluation_output_path folder_ckpt_located