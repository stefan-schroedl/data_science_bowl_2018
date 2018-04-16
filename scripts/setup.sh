conda create -y -n py27 python=2.7 anaconda
source activate py27
conda install -y -c pytorch faiss-cpu 
conda install -y -c menpo opencv3
conda install -y pytorch-cpu torchvision -c pytorch
pip install imgaug imutils tqdm boto3 configargparse
pip install numpy --upgrade
git clone https://github.com/misko/data_science_bowl_2018.git
cd data_science_bowl_2018
git checkout misko2_big_changes
git pull


#python main.py --model knn --data ../../ --stage stage1 -v 0.01 --eval-every 600 --knn-normalize 0 --knn-weird-mean 40  --knn-enhance-its 0
#export OMP_WAIT_POLICY=PASSIVE; OMP_NUM_THREADS=36 python main.py -c eval.cfg --model knn --data ../..// --stage stage2 --eval-every 10 --do submit --resume knn.pkl
