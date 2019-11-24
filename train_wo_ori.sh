#PBS -N TSR-TVD_train_wo_ori
#PBS -l walltime=11:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/TSR-TVD/model
python main.py --root ../exavisData/combustion --save-dir ../saved_models_wo_ori  --volume-loss --sn --wo-ori-volume --batch-size 1 --log-every 15 --test-every 45 --check-every 150 --lr 1e-4 --d-lr 4e-4