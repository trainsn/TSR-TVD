#PBS -N TSR-TVD_infer_66-122 
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/TSR-TVD/eval
cp ../exavisData/combustion/test_cropped/volume_test_list_66-74.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred --infering-step 7
cp ../exavisData/combustion/test_cropped/volume_test_list_74-82.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred --infering-step 7
cp ../exavisData/combustion/test_cropped/volume_test_list_82-90.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred --infering-step 7
cp ../exavisData/combustion/test_cropped/volume_test_list_90-98.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred --infering-step 7
cp ../exavisData/combustion/test_cropped/volume_test_list_98-106.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred --infering-step 7
cp ../exavisData/combustion/test_cropped/volume_test_list_106-114.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred --infering-step 7
cp ../exavisData/combustion/test_cropped/volume_test_list_114-122.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred --infering-step 7

