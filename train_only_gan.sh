#PBS -N TSR-TVD_train_only_GAN
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/TSR-TVD/model
nohup python main.py --root ../exavisData/combustion --save-dir ../saved_models_gan  --gan-loss mse --gan-loss-weight 1 --gen-sn --dis-sn --upsample-mode lr --norm Instance --batch-size 4 --log-every 4 --test-every 60 --check-every 300 --lr 1e-4 --d-lr 4e-4 --volume-train-list volume_train_list.txt --forward --backward > train_gan.log 2>&1 &






