import argparse
import pprint
from typing import Optional
from typing import Sequence
import calamari_ocr
from calamari_ocr.scripts.train import main
from calamari_ocr.scripts.train import parse_args
import tfaip
import logging
import sys
import matplotlib.pyplot as plt


def train(model, epochs, line_height, codec_include):
    plot_path = '../data/Markings/CALAMARI/output/'+model+'/logs/loss_plot.png'
    log_path = '../data/Markings/CALAMARI/output/'+model+'/logs/loss_data.txt'
    train_img_path = '../data/Markings/CALAMARI/dataset/'+model+'/train/*.jpg'
    val_img_path = '../data/Markings/CALAMARI/dataset/'+model+'/validation/*.jpg'
    output_checkpoint_path = '../data/Markings/CALAMARI/models/'+model
    pretrained_model_path = '../data/Markings/CALAMARI/models/pre-trained/best.ckpt'

    out = open(log_path,'w')
    sys.stdout = out
    sys.stderr = out

    logging.getLogger("calamari_ocr").setLevel(logging.WARNING) # eviter que les logs s'affiche dans les outputs
    logging.getLogger("tfaip").setLevel(logging.WARNING) # eviter que les logs s'affiche dans les outputs

    # définir les paramètres de training de calamari (utiliser dir(params.gen.train) pour chercher la syntaxe)
    args = ['--train.images', train_img_path,
            '--val.images', val_img_path,
            '--trainer.epochs', epochs,
            '--trainer.output_dir', output_checkpoint_path,
            '--data.line_height', line_height, 
            '--warmstart.model', pretrained_model_path,
            '--codec.include', codec_include]

    calamari_ocr.scripts.train.main(parse_args(args))

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    out.close()
    fichier = open(log_path, "r")
    lignes = fichier.readlines()
    loss_values = []
    for ligne in lignes:
        if 'p - loss: ' in ligne:
            chain = 'loss/mean_epoch:'
            pos = ligne.find(chain)
            loss_values.append(float(ligne[pos+17:pos+23]))
    plt.plot([i for i in range (len(loss_values))], loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(plot_path)
    plt.show()
    fichier.close()
    

def main_train(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verb', default=False, help='enables/disables messages', type=bool)
    parser.add_argument('--model', default='pre-trained', help='choose your model', type=str)
    parser.add_argument('--epochs', default='1000', help='num of epochs', type=str)
    parser.add_argument('--line_height', default='48', help='num of pixels of the line height', type=str)
    parser.add_argument('--codec_include', default='B R A K E S N P C 0 1 2 3 4 5 6 7 8 9', help='which letters can be detected', type=str)
    args = parser.parse_args(argv)

    if vars(args)['verb'] :
        pprint.pprint(vars(args))

    model, epochs, line_height, codec_include = vars(args)['model'],  vars(args)['epochs'], vars(args)['line_height'], vars(args)['codec_include']
    train(model, epochs, line_height, codec_include)

if __name__ == '__main__':
    exit(main_train())