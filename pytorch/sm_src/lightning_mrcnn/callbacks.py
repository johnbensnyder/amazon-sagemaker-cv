import os
import pickle, gzip
import torch
import pytorch_lightning as pl
from time import time
from pathlib import Path
import shutil
import functools

import smdebug.pytorch as smd
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfig
from smdebug.core.collection import CollectionKeys
from smdebug.core.config_constants import DEFAULT_CONFIG_FILE_PATH

from lightning_mrcnn.coco_eval import mlperf_test_early_exit, image_formatter
from maskrcnn_benchmark.engine.tester import test

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

class COCOEvaluator(pl.Callback):
    
    def on_fit_start(self, trainer, pl_module, stage=None):
        self.step = 0
        self.per_iter_callback_fn = functools.partial(mlperf_test_early_exit,
                                                iters_per_epoch=pl_module.iters_per_epoch,
                                                tester=functools.partial(test, cfg=pl_module.cfg, shapes=pl_module.shapes),
                                                model=pl_module.model,
                                                distributed=pl_module.distributed,
                                                min_bbox_map=pl_module.cfg.MLPERF.MIN_BBOX_MAP,
                                                min_segm_map=pl_module.cfg.MLPERF.MIN_SEGM_MAP,
                                                world_size=pl_module.world_size)
    
    def on_train_batch_start(self, trainer, module, *args, **kwargs):
        self.per_iter_callback_fn(self.step)
        self.step += 1

class CheckpointEveryNSteps(pl.Callback):
    def __init__(
        self,
        output_path="/opt/ml/checkpoints/",
        prefix="MRCNN_checkpoint",
    ):
        """
        Args:
        """
        self.__dict__.update(locals())
        
    def on_fit_start(self, trainer, pl_module, stage=None):
        self.interval = pl_module.iters_per_epoch
        self.step = 0
    
    @pl.utilities.rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _):
        if self.step%self.interval==0 and self.step>0:
            ckpt_path = os.path.join(self.output_path, f"{self.prefix}_step_{self.step}.ckpt")
            trainer.save_checkpoint(ckpt_path)
        self.step += 1
    
class PlSageMakerLogger(pl.Callback):
    
    def __init__(self, frequency=100):
        self.frequency=frequency
    
    def on_train_epoch_start(self, trainer, module, *args, **kwargs):
        self.step_time_start = time()
        self.step = 0
    
    @pl.utilities.rank_zero_only
    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        if self.step%self.frequency==0:
            logs = ["Step: {}".format(self.step),
                    "LR: {0:.4f}".format(float(trainer.model.scheduler.get_lr()[0]))]
            for key,value in trainer.logged_metrics.items():
                logs.append("{0}: {1:.4f}".format(key, float(value)))
            step_time_end = time()
            logs.append("Step time: {0:.2f} milliseconds".format((step_time_end - self.step_time_start)/self.frequency * 1000))
            print(' '.join(logs))
            print("Step : {}".format(self.step))
            print("Training Losses:")
            print(' '.join(["{0}: {1:.4f}".format(key, float(value)) \
                            for key,value in trainer.logged_metrics.items()]))
            step_time_end = time()
            print("Step time: {0:.2f} milliseconds".format((step_time_end - self.step_time_start)/self.frequency * 1000))
            self.step_time_start = step_time_end
        self.step += 1
        
    @pl.utilities.rank_zero_only
    def on_validation_end(self, trainer, module, *args, **kwargs):
        print("Validation")
        print(' '.join(["{0}: {1:.4f}".format(key, float(value)) \
                        for key,value in trainer.logged_metrics.items() if 'val' in key]))

class ProfilerCallback(pl.Callback):
    
    def __init__(self, start_step=200, num_steps=25, output_dir='/opt/ml/checkpoints/profiler/'):
        super().__init__()
        self.__dict__.update(locals())
        self.step = 0
        self.profiler = torch.profiler.profile(activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                                    schedule=torch.profiler.schedule(wait=5,
                                                                     warmup=5,
                                                                     active=self.num_steps),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, "tensorboard")),
                                    with_stack=True)
    
    def on_train_batch_start(self, trainer, module, *args, **kwargs):
        if self.step==self.start_step:
            self.profiler.__enter__()
    
    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        if self.step>=self.start_step and self.step<=self.start_step + self.num_steps:
            self.profiler.step()
        if self.step==self.start_step + self.num_steps:
            self.profiler.__exit__(None, None, None)
        self.step += 1

class SMDebugCallback(pl.Callback):
    def __init__(self, out_dir='/opt/ml/code/smdebugger',
                       log_frequency=10,
                       export_tensorboard=True,
                       tensorboard_dir=None,
                       dry_run=False,
                       reduction_config=ReductionConfig(['mean']),
                       save_config=SaveConfig(save_interval=25),
                       include_regex=None,
                       include_collections=[CollectionKeys.LOSSES],
                       save_all=False,
                       include_workers="one",
                    ):
        super().__init__()
        self.__dict__.update(locals())
        if self.out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)
            assert not Path(self.out_dir).exists()
        
    def on_fit_start(self, trainer, pl_module, stage=None):
        self.step = 0
        if Path(DEFAULT_CONFIG_FILE_PATH).exists():
            # smd.Hook.register_hook(pl_module.model, pl_module.criterion)
            self.hook = smd.Hook.create_from_json_file()
        else:
            self.hook = smd.Hook(out_dir=self.out_dir,
                            export_tensorboard=self.export_tensorboard,
                            tensorboard_dir=self.tensorboard_dir,
                            reduction_config=self.reduction_config,
                            save_config=self.save_config,
                            include_regex=self.include_regex,
                            include_collections=self.include_collections,
                            save_all=self.save_all,
                            include_workers=self.include_workers)
        self.hook.register_module(pl_module.model)
        # hook.register_loss(pl_module.criterion)
        
    @pl.utilities.rank_zero_only
    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        if self.step%self.log_frequency==0:
            for key, value in trainer.logged_metrics.items():
                self.hook.save_scalar(key, value)
        if self.step%100==0:
            for i, image in enumerate(module.images.tensors):
                self.hook.save_tensor(f'image_detections_{i}', 
                                      image_formatter(image, module.detections[i]), 
                                      'image', dataformats='hwc')
        self.step += 1