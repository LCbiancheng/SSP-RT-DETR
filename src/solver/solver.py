"""by lyuwenyu
"""

import copy

import torch 

from datetime import datetime
from pathlib import Path 
from typing import Dict

from src.misc import dist, is_concise_logging, suppress_output
from src.core import BaseConfig


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        
        self.cfg = cfg 

    def setup(self, ):
        '''Avoid instantiating unnecessary classes 
        '''
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch
        self._init_training_state()

        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor

        # NOTE (lvwenyu): should load_tuning_state before ema instance building
        if self.cfg.tuning:
            if not is_concise_logging():
                print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None 

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def _init_training_state(self):
        self.best_stat = {'epoch': -1}

        early_stopping_cfg = copy.deepcopy(getattr(self.cfg, 'early_stopping', {}) or {})
        enabled = bool(early_stopping_cfg.get('enabled', False))
        mode = str(early_stopping_cfg.get('mode', 'max')).lower()
        patience = int(early_stopping_cfg.get('patience', 10))
        start_epoch = int(early_stopping_cfg.get('start_epoch', 0))

        if mode not in ('min', 'max'):
            raise ValueError(f"Unsupported early_stopping.mode: {mode}")
        if enabled and patience < 1:
            raise ValueError('early_stopping.patience must be >= 1 when enabled')
        if start_epoch < 0:
            raise ValueError('early_stopping.start_epoch must be >= 0')

        self.early_stopping = {
            'enabled': enabled,
            'monitor': str(early_stopping_cfg.get('monitor', 'mAP_50:95')),
            'mode': mode,
            'patience': patience,
            'min_delta': float(early_stopping_cfg.get('min_delta', 0.0)),
            'start_epoch': start_epoch,
            'best_score': None,
            'best_epoch': -1,
            'bad_epochs': 0,
            'stopped_epoch': -1,
        }


    def train(self, ):
        self.setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        # NOTE instantiating order
        if self.cfg.resume:
            if not is_concise_logging():
                print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        self.train_dataloader = dist.warp_loader(self.cfg.train_dataloader, \
            shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)


    def eval(self, ):
        self.setup()
        eval_loader = getattr(self.cfg, 'test_dataloader', None)
        if eval_loader is None:
            eval_loader = self.cfg.val_dataloader

        self.val_dataloader = dist.warp_loader(eval_loader, \
            shuffle=eval_loader.shuffle)

        if self.cfg.resume:
            if not is_concise_logging():
                print(f'resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)


    def state_dict(self, last_epoch):
        '''state dict
        '''
        state = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()

        # TODO
        state['last_epoch'] = last_epoch

        if getattr(self, 'criterion', None) is not None:
            state['criterion'] = self.criterion.state_dict()

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            # state['last_epoch'] = self.lr_scheduler.last_epoch

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        if hasattr(self, 'best_stat'):
            state['best_stat'] = copy.deepcopy(self.best_stat)

        if hasattr(self, 'early_stopping'):
            state['early_stopping_state'] = {
                'best_score': self.early_stopping.get('best_score', None),
                'best_epoch': self.early_stopping.get('best_epoch', -1),
                'bad_epochs': self.early_stopping.get('bad_epochs', 0),
                'stopped_epoch': self.early_stopping.get('stopped_epoch', -1),
            }

        return state


    def load_state_dict(self, state):
        '''load state dict
        '''
        concise_mode = is_concise_logging()
        # TODO
        if hasattr(self, 'last_epoch') and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            if not concise_mode:
                print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            if not concise_mode:
                print('Loading model.state_dict')

        if getattr(self, 'criterion', None) and 'criterion' in state:
            incompatible = self.criterion.load_state_dict(state['criterion'], strict=False)
            if not concise_mode:
                missing = getattr(incompatible, 'missing_keys', [])
                unexpected = getattr(incompatible, 'unexpected_keys', [])
                if missing or unexpected:
                    print(f'Loading criterion.state_dict with missing={missing}, unexpected={unexpected}')
                else:
                    print('Loading criterion.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            if not concise_mode:
                print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            if not concise_mode:
                print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            if not concise_mode:
                print('Loading lr_scheduler.state_dict')
            try:
                cfg_ms = getattr(self.cfg, 'yaml_cfg', {}).get('lr_scheduler', {}).get('milestones', [])
                if hasattr(self.lr_scheduler, 'milestones') and cfg_ms:
                    existing = set(self.lr_scheduler.milestones)
                    for m in cfg_ms:
                        if m not in existing:
                            self.lr_scheduler.milestones.append(m)
                    self.lr_scheduler.milestones.sort()
            except Exception:
                pass

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            if not concise_mode:
                print('Loading scaler.state_dict')

        if hasattr(self, 'best_stat') and 'best_stat' in state:
            self.best_stat = state['best_stat']
            if not concise_mode:
                print('Loading best_stat')

        if hasattr(self, 'early_stopping'):
            early_stopping_state = state.get('early_stopping_state', None)
            if early_stopping_state is not None:
                self.early_stopping['best_score'] = early_stopping_state.get('best_score', None)
                self.early_stopping['best_epoch'] = early_stopping_state.get('best_epoch', -1)
                self.early_stopping['bad_epochs'] = early_stopping_state.get('bad_epochs', 0)
                self.early_stopping['stopped_epoch'] = early_stopping_state.get('stopped_epoch', -1)
                if not concise_mode:
                    print('Loading early_stopping_state')


    def save(self, path):
        '''save state
        '''
        state = self.state_dict(getattr(self, 'last_epoch', -1))
        dist.save_on_master(state, path)


    def resume(self, path):
        '''load resume
        '''
        # for cuda:0 memory
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def load_tuning_state(self, path,):
        """only load model for tuning and skip missed/dismatched keys
        """
        concise_mode = is_concise_logging()
        if 'http' in path:
            with suppress_output(enabled=concise_mode):
                state = torch.hub.load_state_dict_from_url(
                    path,
                    map_location='cpu',
                    progress=not concise_mode,
                )
        else:
            state = torch.load(path, map_location='cpu')

        module = dist.de_parallel(self.model)
        
        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        if not concise_mode:
            print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}


    def fit(self, ):
        raise NotImplementedError('')

    def val(self, ):
        raise NotImplementedError('')
