#PBS -N TSR-TVD_infer_90-122 
#PBS -l walltime=6:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/TSR-TVD/eval
cp ../exavisData/combustion/test_cropped/volume_test_list_90-94.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred
cp ../exavisData/combustion/test_cropped/volume_test_list_94-98.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred
cp ../exavisData/combustion/test_cropped/volume_test_list_98-102.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred
cp ../exavisData/combustion/test_cropped/volume_test_list_102-106.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred
cp ../exavisData/combustion/test_cropped/volume_test_list_106-110.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred
cp ../exavisData/combustion/test_cropped/volume_test_list_110-114.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred
cp ../exavisData/combustion/test_cropped/volume_test_list_114-118.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred
cp ../exavisData/combustion/test_cropped/volume_test_list_118-122.txt ../exavisData/combustion/test_cropped/volume_test_list.txt
python eval.py --root ../exavisData/combustion --resume ../saved_models/model_5_250_pth.tar --save-pred ../save_pred
