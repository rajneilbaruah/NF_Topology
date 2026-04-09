$DATAPATH = /home/work/datasets/NFSpheres/eemumu_mup.json

python train_all.py --data $DATAPATH--seed 321

python plot_all.py --data $DATAPATH 
 
python analyse_all.py --data $DATAPATH
