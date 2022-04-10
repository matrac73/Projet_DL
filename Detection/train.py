from __future__ import print_function
import argparse
import pprint
from typing import Optional
from typing import Sequence
import os
from east import multigpu_train

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

def train():
    multigpu_train.main()

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verb', default=False, help='enables/disables messages', type=bool)
    parser.add_argument('--gpu_list', default=0, help='', type=int)
    parser.add_argument('--input_size', default=512, help='', type=int)
    parser.add_argument('--batch_size_per_gpu', default=1, help='', type=int)
    parser.add_argument('--checkpoint_path', default='', help='', type=str)
    parser.add_argument('--text_scale', default=512, help='', type=int)
    parser.add_argument('--training_data_path', default='', help='', type=str)
    parser.add_argument('--geometry', default='RBOX', help='', type=str)
    parser.add_argument('--learning_rate', default=0.0001, help='', type=float)
    parser.add_argument('--num_readers', default=24, help='', type=int)
    parser.add_argument('--pretrained_model_path', default='', help='', type=str)
    parser.add_argument('--path_training_logs', default='', help='', type=str)
    args = parser.parse_args(argv)

    if vars(args)['verb'] :
        pprint.pprint(vars(args))

    setup(vars(args)['verb'])
    train()

if __name__ == '__main__':
    exit(main())