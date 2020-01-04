#PBS -N TSR-TVD_train_step9
#PBS -l walltime=47:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/TSR-TVD/model
python main.py --root ../exavisData/combustion --save-dir ../saved_models_step3  --volume-loss --gen-sn --dis-sn --upsample-mode lr --norm Instance --batch-size 2 --log-every 4 --test-every 60 --check-every 400 --lr 1e-4 --d-lr 4e-4 --volume-train-list volume_train_list_step3.txt --volume-test-list volume_test_list_step3.txt --forward --backward --training-step 3 --resume ../saved_models_mse_ins/model_2_800_pth.tar