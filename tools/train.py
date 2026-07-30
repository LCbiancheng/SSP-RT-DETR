"""by lyuwenyu
"""

import os 
import sys 

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    if not args.test_only:
        os.environ.setdefault('ABLATION_CONCISE', '1')

    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg_kwargs = {}
    if args.resume is not None:
        cfg_kwargs['resume'] = args.resume
    if args.tuning is not None:
        cfg_kwargs['tuning'] = args.tuning
    if args.amp is not None:
        cfg_kwargs['use_amp'] = args.amp

    cfg = YAMLConfig(args.config, **cfg_kwargs)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.set_defaults(amp=None)
    parser.add_argument('--amp', dest='amp', action='store_true',
                        help='Override config to enable AMP.')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='Override config to disable AMP.')
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)
