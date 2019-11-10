#PBS -N TSR-TVD train 
#PBS -l walltime=11:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/TSR-TVD/model 
python main.py --root ../exavisData/combustion --save-dir ../saved_models  --gan-loss mse --volume-loss --feature-loss --sn  --resume ../saved_models/model_3_130_pth.tar --batch-size 1 --log-every 15 --test-every 45 --check-every 150 --lr 1e-5 --d-lr 4e-5




