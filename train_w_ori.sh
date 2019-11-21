#PBS -N TSR-TVD_train_w_ori
#PBS -l walltime=7:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/TSR-TVD/model
python main.py --root ../exavisData/combustion --save-dir ../saved_models_mse  --volume-loss --sn --batch-size 1 --log-every 15 --test-every 45 --check-every 150 --lr 1e-4 --d-lr 4e-4





