'''
by lyuwenyu
'''
import json
import csv
import re
import shutil

import torch

from src.misc import dist, is_concise_logging
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    EARLY_STOP_METRIC_ALIASES = {
        'mAP_50:95': ('coco_eval_bbox', 0),
        'mAP_50': ('coco_eval_bbox', 1),
        'mAP_75': ('coco_eval_bbox', 2),
        'AP_small': ('coco_eval_bbox', 3),
        'AP_medium': ('coco_eval_bbox', 4),
        'AP_large': ('coco_eval_bbox', 5),
    }

    def _resolve_early_stopping_metric(self, test_stats):
        monitor = self.early_stopping['monitor']

        if monitor in self.EARLY_STOP_METRIC_ALIASES:
            metric_key, metric_index = self.EARLY_STOP_METRIC_ALIASES[monitor]
        elif monitor.endswith(']') and '[' in monitor:
            metric_key, raw_index = monitor[:-1].split('[', 1)
            metric_index = int(raw_index)
        elif monitor in test_stats:
            value = test_stats[monitor]
            if isinstance(value, (list, tuple)):
                raise ValueError(
                    f"Early stopping monitor '{monitor}' resolves to a sequence. "
                    "Use a bracket index such as coco_eval_bbox[0]."
                )
            return float(value)
        else:
            supported = ', '.join(sorted(self.EARLY_STOP_METRIC_ALIASES.keys()))
            raise KeyError(
                f"Unsupported early stopping monitor '{monitor}'. "
                f"Supported aliases: {supported}, or use metrics like coco_eval_bbox[0]."
            )

        value = test_stats.get(metric_key, None)
        if value is None:
            raise KeyError(f"Metric '{metric_key}' not found in evaluation stats.")
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Metric '{metric_key}' is not indexable, but '{monitor}' expects an index.")
        if metric_index >= len(value):
            raise IndexError(f"Metric '{metric_key}' index {metric_index} is out of range.")

        return float(value[metric_index])

    @staticmethod
    def _is_metric_improved(current, best, mode, min_delta):
        if best is None:
            return True
        if mode == 'max':
            return current > (best + min_delta)
        return current < (best - min_delta)

    def _update_early_stopping(self, metric_value, epoch):
        early_stopping = self.early_stopping
        if not early_stopping['enabled']:
            return False, False

        improved = self._is_metric_improved(
            metric_value,
            early_stopping.get('best_score', None),
            early_stopping['mode'],
            early_stopping['min_delta'],
        )

        if improved:
            early_stopping['best_score'] = float(metric_value)
            early_stopping['best_epoch'] = epoch
            early_stopping['bad_epochs'] = 0
            early_stopping['stopped_epoch'] = -1
            return False, True

        if (epoch + 1) < early_stopping['start_epoch']:
            return False, False

        early_stopping['bad_epochs'] += 1
        should_stop = early_stopping['bad_epochs'] >= early_stopping['patience']
        if should_stop:
            early_stopping['stopped_epoch'] = epoch

        return should_stop, False

    def _resolve_result_csv_name(self):
        configured_name = getattr(self.cfg, 'result_csv_name', None)
        if configured_name:
            return configured_name

        output_name = getattr(self.output_dir, 'name', '') or 'experiment'
        safe_name = re.sub(r'[^0-9A-Za-z]+', '_', output_name).strip('_').lower()
        if not safe_name:
            safe_name = 'experiment'
        return f'{safe_name}_results.csv'

    def fit(self, ):
        concise_mode = is_concise_logging()
        if not concise_mode:
            print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if not concise_mode:
            print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = self.best_stat

        csv_path = None
        summary_path = None
        result_csv_name = self._resolve_result_csv_name()
        num_classes = self.cfg.yaml_cfg.get('num_classes', 6)
        csv_header = [
            'Epoch', 'mAP_50:95', 'mAP_50', 'mAP_75',
            'AP_small', 'AP_medium', 'AP_large',
            'FPS', 'Params(M)',
        ] + [f'Class_{i}_AP' for i in range(num_classes)]
        if self.output_dir and dist.is_main_process():
            csv_path = self.output_dir / result_csv_name
            summary_path = self.output_dir / 'summary.json'
            if not (self.cfg.resume and csv_path.exists()):
                with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_header)
            display_header = ['Epoch/Total', 'Progress', *csv_header]
            print('\t'.join(display_header))

        def _progress_bar(cur_epoch, total_epoches, width=28):
            total = max(1, int(total_epoches))
            done = max(1, min(int(cur_epoch), total))
            ratio = float(done) / float(total)
            fill = int(round(width * ratio))
            bar = '#' * fill + '-' * (width - fill)
            return f'[{bar}] {ratio * 100:5.1f}%'

        def _format_epoch_metric_line(epoch_disp, total_epoches, detail, ordered_class_aps):
            metric_values = [
                f'{epoch_disp}/{max(1, int(total_epoches))}',
                _progress_bar(epoch_disp, total_epoches),
                str(epoch_disp),
                f"{detail['mAP_50_95']:.4f}",
                f"{detail['mAP_50']:.4f}",
                f"{detail['mAP_75']:.4f}",
                f"{detail['AP_S']:.4f}",
                f"{detail['AP_M']:.4f}",
                f"{detail['AP_L']:.4f}",
                f"{detail['FPS']:.1f}",
                f"{detail['Params(M)']:.2f}",
            ]
            metric_values.extend(f'{class_ap:.4f}' for class_ap in ordered_class_aps)
            return '\t'.join(metric_values)

        def _write_summary(epoch_disp, detail):
            if summary_path is None or not dist.is_main_process():
                return
            summary = {
                'best_epoch': int(best_stat.get('epoch', -1)) + 1 if best_stat.get('epoch', -1) >= 0 else -1,
                'best_mAP_50:95': float(best_stat.get('coco_eval_bbox', 0.0)),
                'last_epoch': int(epoch_disp),
                'last_mAP_50:95': float(detail['mAP_50_95']),
                'last_mAP_50': float(detail['mAP_50']),
                'last_mAP_75': float(detail['mAP_75']),
                'fps': float(detail['FPS']),
                'params_million': float(detail['Params(M)']),
                'n_parameters': int(n_parameters),
                'stopped_early': bool(self.early_stopping.get('stopped_epoch', -1) >= 0),
                'early_stopping': {
                    'enabled': bool(self.early_stopping.get('enabled', False)),
                    'monitor': self.early_stopping.get('monitor'),
                    'mode': self.early_stopping.get('mode'),
                    'patience': int(self.early_stopping.get('patience', 0)),
                    'min_delta': float(self.early_stopping.get('min_delta', 0.0)),
                    'start_epoch': int(self.early_stopping.get('start_epoch', 0)),
                    'bad_epochs': int(self.early_stopping.get('bad_epochs', 0)),
                    'best_epoch': int(self.early_stopping.get('best_epoch', -1)) + 1 if self.early_stopping.get('best_epoch', -1) >= 0 else -1,
                    'best_score': None if self.early_stopping.get('best_score', None) is None else float(self.early_stopping['best_score']),
                    'stopped_epoch': int(self.early_stopping.get('stopped_epoch', -1)) + 1 if self.early_stopping.get('stopped_epoch', -1) >= 0 else -1,
                },
            }
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        def _prune_output_dir():
            if not self.output_dir or not dist.is_main_process():
                return
            allowed = {'checkpoint.pth', 'best.pth', 'summary.json', result_csv_name, 'experiment_results.csv'}
            for path in self.output_dir.iterdir():
                if path.name in allowed:
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)

        stopped_early = False

        if hasattr(self.criterion, '_total_epochs'):
            self.criterion._total_epochs = args.epoches

        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if hasattr(self.train_dataloader, 'dataset') and hasattr(self.train_dataloader.dataset, 'set_mosaic_epoch'):
                self.train_dataloader.dataset.set_mosaic_epoch(epoch, args.epoches)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
                self.output_dir,
                result_csv_name=result_csv_name,
            )

            stop_training = False
            early_stop_metric = None
            if self.early_stopping['enabled']:
                early_stop_metric = self._resolve_early_stopping_metric(test_stats)
                stop_training, _ = self._update_early_stopping(early_stop_metric, epoch)

            is_new_best = False
            for k in test_stats.keys():
                if k == 'AP_detail':
                    continue
                val = test_stats[k][0] if isinstance(test_stats[k], (list, tuple)) else test_stats[k]
                if k in best_stat:
                    if val > best_stat[k]:
                        is_new_best = True
                        best_stat['epoch'] = epoch
                        best_stat[k] = val
                else:
                        is_new_best = True
                        best_stat['epoch'] = epoch
                        best_stat[k] = val

            if self.output_dir and dist.is_main_process():
                if is_new_best:
                    dist.save_on_master(self.state_dict(epoch), self.output_dir / 'best.pth')
                dist.save_on_master(self.state_dict(epoch), self.output_dir / 'checkpoint.pth')

            if 'AP_detail' in test_stats:
                detail = test_stats['AP_detail']
                detail['Params(M)'] = float(n_parameters) / 1e6
                epoch_disp = epoch + 1
                class_aps = detail.get('Class_APs', {})
                ordered_class_aps = [float(class_aps.get(i, 0.0)) for i in range(num_classes)]

                epoch_line = _format_epoch_metric_line(epoch_disp, args.epoches, detail, ordered_class_aps)
                print(epoch_line)
                if csv_path is not None and dist.is_main_process():
                    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        row = [
                            epoch_disp,
                            f"{detail['mAP_50_95']:.4f}",
                            f"{detail['mAP_50']:.4f}",
                            f"{detail['mAP_75']:.4f}",
                            f"{detail['AP_S']:.4f}",
                            f"{detail['AP_M']:.4f}",
                            f"{detail['AP_L']:.4f}",
                            f"{detail['FPS']:.1f}",
                            f"{detail['Params(M)']:.2f}",
                        ]
                        row.extend([f"{v:.4f}" for v in ordered_class_aps])
                        writer.writerow(row)
                _write_summary(epoch_disp, detail)

            if 'AP_detail' in test_stats:
                del test_stats['AP_detail']

            if stop_training:
                stopped_early = True
                if not concise_mode:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}: "
                        f"{self.early_stopping['monitor']} did not improve for "
                        f"{self.early_stopping['patience']} consecutive epochs."
                    )
                break

        _prune_output_dir()
        if stopped_early and not concise_mode:
            best_epoch = self.early_stopping.get('best_epoch', -1)
            best_score = self.early_stopping.get('best_score', None)
            if best_epoch >= 0 and best_score is not None:
                print(
                    f"Early stopped. Best {self.early_stopping['monitor']}={best_score:.4f} "
                    f"at epoch {best_epoch + 1}."
                )

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            base_ds,
            self.device,
            self.output_dir,
            result_csv_name=self._resolve_result_csv_name(),
        )

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
