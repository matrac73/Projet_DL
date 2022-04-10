import argparse
import pprint
from typing import Optional
from typing import Sequence
import calamari_ocr
import logging
from calamari_ocr import __version__
from paiargparse import PAIArgumentParser
from calamari_ocr.scripts.predict import PredictArgs
from calamari_ocr.scripts.predict import run
from calamari_ocr.scripts.eval import main
from calamari_ocr.scripts.eval import parse_args
from shutil import copyfile
import os
import sys

def test(model):
    logging.getLogger("calamari_ocr").setLevel(logging.WARNING) # eviter que les logs s'affiche dans les outputs

    test_img_path = '../data/Markings/CALAMARI/dataset/test-set/*.jpg'
    output_img_path = '../data/Markings/CALAMARI/output/'+model+'/predictions'
    checkpoint_path = '../data/Markings/CALAMARI/models/'+model+'/best.ckpt'

    args = ['--data.images', test_img_path,
            '--checkpoint', checkpoint_path,
            '--output_dir', output_img_path]

    parser = PAIArgumentParser()
    parser.add_argument("--version", action="version", version="%(prog)s v" + __version__)
    parser.add_root_argument("root", PredictArgs, flat=True)
    args = parser.parse_args(args)
    run(args.root)

def evaluate(model):
    path = '../data/Markings/CALAMARI/dataset/test-set'
    gt_path = '../data/Markings/CALAMARI/dataset/test-set/*.txt'
    log_path = '../data/Markings/CALAMARI/output/'+model+'/logs/inference_statistics.txt'

    out = open(log_path,'w')
    sys.stdout = out
    sys.stderr = out

    logging.getLogger("calamari_ocr").setLevel(logging.WARNING) # eviter que les logs s'affiche dans les outputs

    # Déplacer les fichiers gt dans output
    for file_path in os.listdir(path):
        if file_path.endswith('.gt.txt'):
            destination = '../data/Markings/CALAMARI/output/'+model+'/predictions/' + file_path[-46:]
            copyfile(path + '/' +file_path, destination)

    # Définir les paramètre de training de calamari (utiliser dir(params.gen.train) pour chercher la syntaxe)
    args = ['--gt.texts', '../data/Markings/CALAMARI/output/'+model+'/predictions/*.gt.txt']

    main(parse_args(args))

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    out.close()

def main_test(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verb', default=False, help='enables/disables messages', type=bool)
    parser.add_argument('--model', default='pre-trained', help='choose the model to test', type=str)
    args = parser.parse_args(argv)

    if vars(args)['verb'] :
        pprint.pprint(vars(args))

    test(vars(args)['model'])
    evaluate(vars(args)['model'])

if __name__ == '__main__':
    exit(main_test())