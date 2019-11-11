#PBS -N TSR-TVD_infer_50-54 
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/TSR-TVD/eval
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred



