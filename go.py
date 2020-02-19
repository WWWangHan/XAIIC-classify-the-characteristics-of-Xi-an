import os

os.system("python src/main.py --data_url 'crawl-data' --train_url 'crawl-data/train_val/train' --data_local 'crawl-data/train_val/' --deploy_script_path 'src_v2_20191120/deploy_scripts' --arch 'resnet50' --pretrained True --num_classes 54 --workers 4 --epochs 100 --seed 0 --batch_size 32 --lr 0.005 --last_fc_out 256 --gpu 0")
