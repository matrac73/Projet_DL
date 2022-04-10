from __future__ import print_function
import argparse
import pprint
from typing import Optional
from typing import Sequence
import os
import glob
from east import sleipnir

def setup(verb):
    # Moving to test data folder
    if 'paths_set' not in locals():
        paths_set = 0
    if not paths_set:
        project_path = os.getcwd() # current working directory
        paths_set = 1
    else:
        if verb:
            print('Paths already set')
    if verb:
        print('Project root path ', project_path)

def test(model):

    engine = sleipnir.sleipnir(verbose=True, logging=False, write_output_images=False) # initializing OCR engine

    checkpoint_path = '../data/Markings/EAST/models/' + model # model to use for detection
    if model == 'pre-trained':
        checkpoint_path = '../data/Markings/EAST/models/pre-trained-test'
    output_path = '../data/Markings/EAST/output/' + model + '/predictions' # folder path where output is stored
    stat_save_path = '../data/Markings/EAST/output/' + model + '/logs/inference_statistics.txt' # folder path where statistics are stored
    test_set_path = '../data/Markings/EAST/dataset/test-set' # folder path where true labels are stored
    eval_script_path = './east/eval.py' # path to script used vor evaluation

    engine.get_true_labels(test_set_path) # setting path to true labels
    engine.get_input_images(test_set_path)
    engine.get_eval_script(eval_script_path) # setting path for evaluation script
    engine.get_model_checkpoint(checkpoint_path) # setting model checkpoint
    engine.get_output_dir(output_path) # setting output path
    engine.process_images() # process images
    engine.process_results() # filter results and add boxes to pictures

    try:
        os.remove(stat_save_path)
    except:
        print("Warning : penser à bien faire fichier_stat.close()")
        
    fichier_stat = open(stat_save_path, "a")
    liste_stat = list(engine.compute_average_error())
    fichier_stat.write("Average error : " + str(liste_stat[0]) + "\n")
    fichier_stat.write("Partial detection rate : " + str(round(liste_stat[1]*100,2)) + "\n")
    fichier_stat.write("Full detection rate : " + str(round(liste_stat[2]*100,2)))
    fichier_stat.close()

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verb', default=False, help='enables/disables messages', type=bool)
    parser.add_argument('--model', default='pre-trained', help='choose your model', type=str)
    args = parser.parse_args(argv)

    if vars(args)['verb'] :
        pprint.pprint(vars(args))

    setup(vars(args)['verb'])
    test(vars(args)['model'])

if __name__ == '__main__':
    exit(main())