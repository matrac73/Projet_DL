# EAST - Detection

## Setup du projet

1 . Télécharger **anaconda** et ajouter **conda** au PATH (si ce n'est pas déja fait) 
(il y a des tuto en ligne : https://developers.google.com/earth-engine/guides/python_install-conda par exemple)

2 . Créer un environement virtuel anaconda avec la bonne version de python :
``` bash
conda create -n EAST_venv python=3.6.13
```
3 . Activer l'environement virtuel :
``` bash
conda activate EAST_venv
```
4 . Se placer dans le répertoire dans lequel on veut insaller le dépot git :
``` bash
cd ./path/to/repository/
```
5 . Cloner le dépot git :
``` bash
git clone adresse_git
```
6 . Se placer dans le répertoire qui viens d'être crée :
``` bash
cd ./sls_sleipnir_det.git/
```
7 . Télécharger les dépendances :
``` bash
python -m pip install -r requirements.txt
python -m pip install .\east\Shapely-1.7.1-cp36-cp36m-win_amd64.whl --force-reinstall
```
8 . Faire tourner les modules scripts/train.py et scripts/test.py
``` bash
python train.py --gpu_list 0 --input_size 512 --batch_size_per_gpu 1 --checkpoint_path ../data/Markings/EAST/models/50_images/ --text_scale 512 --training_data_path ../data/Markings/EAST/dataset/50_images/train/ --geometry RBOX --learning_rate 0.0001 --num_readers 24 --pretrained_model_path ../data/Markings/EAST/models/pre-trained-train/resnet_v1_50.ckpt --path_training_logs ../data/Markings/EAST/output/50_images/logs/
``` 
``` bash
python test.py --verb True --model '50_images'
```
