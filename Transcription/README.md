# CALAMARI - Transcription

## Setup du projet

1 . Télécharger **anaconda** et ajouter **conda** au PATH (si ce n'est pas déja fait) 
(il y a des tuto en ligne : https://developers.google.com/earth-engine/guides/python_install-conda par exemple)

2 . Créer un environement virtuel anaconda avec la bonne version de python :
``` bash
conda create -n CALAMARI_venv python=3.7.10
```
3 . Activer l'environement virtuel :
``` bash
conda activate CALAMARI_venv
```
4 . Se placer dans le répertoire dans lequel on veut insaller le dépot git :
``` bash
cd ./path/to/repository/
```
5 . Cloner le dépot git :
``` bash
git clone ssh://***ID***@***IP***/opt/git/sls_sleipnir_tr.git
```
6 . Se placer dans le répertoire qui viens d'être crée :
``` bash
cd ./sls_sleipnir_tr.git/
```
7 . Télécharger les dépendances :
``` bash
python -m pip install -r requirements.txt
```
8 . Faire tourner les modules scripts/train.py et scripts/test.py
``` bash
python train.py --verb True --model '50_images' --epochs '1000' --line_height '48' --codec_include 'B R A K E S N P C 0 1 2 3 4 5 6 7 8 9'
``` 
``` bash
python test.py --verb True --model 'pre-trained'
```
