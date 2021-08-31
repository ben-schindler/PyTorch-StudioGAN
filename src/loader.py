# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/loader.py


from os.path import dirname, abspath, exists, join
import glob
import json
import os
import random
import warnings

from torchlars import LARS
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist

from data_util import Dataset_
from metrics.inception_net import InceptionV3
from sync_batchnorm.batchnorm import convert_model
from worker import WORKER
import utils.log as log
import utils.losses as losses
import utils.ckpt as ckpt
import utils.misc as misc
import models.model as model
import metrics.preparation as pp


def load_worker(local_rank, cfgs, gpus_per_node, run_name, hdf5_path):
    # -----------------------------------------------------------------------------
    # define default variables for loading ckpt or evaluating the trained GAN model.
    # -----------------------------------------------------------------------------
    ada_p, step, best_step, best_fid, best_ckpt_path, is_best = None, 0, 0, None, None, False
    mu, sigma, eval_model, nrow, ncol = None, None, None, 10, 8

    # -----------------------------------------------------------------------------
    # initialize all processes and identify the local rank.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        global_rank = cfgs.RUN.cn*(gpus_per_node) + local_rank
        print("Use GPU: {global_rank} for training.".format(global_rank=global_rank))
        misc.setup(global_rank, cfgs.OPTIMIZATION.world_size)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    # -----------------------------------------------------------------------------
    # define tensorflow writer and python logger.
    # -----------------------------------------------------------------------------
    if local_rank == 0:
        writer = SummaryWriter(log_dir=join("./logs", run_name))
        logger = log.make_logger(run_name, None)
        logger.info("Run name : {run_name}".format(run_name=run_name))
        for k, v in cfgs.super_cfgs.items():
            logger.info("cfgs." + k + " =")
            logger.info(json.dumps(vars(v), indent=2))
    else:
        writer, logger = None, None

    # -----------------------------------------------------------------------------
    # load train and evaluation dataset.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.train:
        if local_rank == 0: logger.info("Load {name} train dataset.".format(name=cfgs.DATA.name))
        train_dataset = Dataset_(data_name=cfgs.DATA.name,
                                 data_path=cfgs.DATA.path,
                                 train=True,
                                 crop_long_edge=cfgs.PRE.crop_long_edge,
                                 resize_size=cfgs.PRE.resize_size,
                                 random_flip=cfgs.PRE.apply_rflip,
                                 hdf5_path=hdf5_path,
                                 load_data_in_memory=cfgs.RUN.load_data_in_memory)
        if local_rank == 0: logger.info("Train dataset size: {dataset_size}".format(dataset_size=len(train_dataset)))
    else:
        train_dataset = None

    if cfgs.RUN.eval + cfgs.RUN.k_nearest_neighbor + cfgs.RUN.frequency_analysis + cfgs.RUN.tsne_analysis:
        if local_rank == 0: logger.info("Load {name} {ref} datasets.".format(name=cfgs.DATA.name, ref=cfgs.RUN.ref_dataset))
        eval_dataset = Dataset_(data_name=cfgs.DATA.name,
                                data_path=cfgs.DATA.path,
                                train=True if cfgs.RUN.ref_dataset == "train" else False,
                                crop_long_edge=False if cfgs.DATA in ["CIFAR10", "CIFAR100", "Tiny_ImageNet"] else True,
                                resize_size=None if cfgs.DATA in ["CIFAR10", "CIFAR100", "Tiny_ImageNet"] else cfgs.DATA.img_size,
                                random_flip=False,
                                hdf5_path=None,
                                load_data_in_memory=False)
        if local_rank == 0: logger.info("Eval dataset size: {dataset_size}".format(dataset_size=len(eval_dataset)))
    else:
        eval_dataset = None

    # -----------------------------------------------------------------------------
    # define a distributed sampler for DDP training.
    # define dataloaders for train and evaluation.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        train_sampler = DistributedSampler(train_dataset)
        cfgs.OPTIMIZATION.batch_size = cfgs.OPTIMIZATION.batch_size//cfgs.OPTIMIZATION.world_size
    else:
        train_sampler = None
    cfgs.OPTIMIZATION.basket_size = cfgs.OPTIMIZATION.batch_size*cfgs.OPTIMIZATION.acml_steps*cfgs.OPTIMIZATION.d_updates_per_step

    if cfgs.RUN.train:
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=cfgs.OPTIMIZATION.basket_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=True,
                                      num_workers=cfgs.RUN.num_workers,
                                      sampler=train_sampler,
                                      drop_last=True)
    else:
        train_dataloader = None

    if cfgs.RUN.eval + cfgs.RUN.k_nearest_neighbor + cfgs.RUN.frequency_analysis + cfgs.RUN.tsne_analysis:
        eval_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=cfgs.OPTIMIZATION.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=cfgs.RUN.num_workers,
                                     drop_last=False)
    else:
        eval_dataloader = None

    # -----------------------------------------------------------------------------
    # load a generator and a discriminator
    # if cfgs.MODEL.apply_g_ema is True, load an exponential moving average generator (Gen_copy).
    # -----------------------------------------------------------------------------
    Gen, Dis, Gen_ema, ema = model.load_generator_discriminator(DATA=cfgs.DATA,
                                                                OPTIMIZATION=cfgs.OPTIMIZATION,
                                                                MODEL=cfgs.MODEL,
                                                                MODULES=cfgs.MODULES,
                                                                RUN=cfgs.RUN,
                                                                device=local_rank,
                                                                logger=logger)

    # -----------------------------------------------------------------------------
    # define optimizers for adversarial training
    # -----------------------------------------------------------------------------
    cfgs.define_optimizer(Gen, Dis)

    # -----------------------------------------------------------------------------
    # load the generator and the discriminator from a checkpoint if possible
    # -----------------------------------------------------------------------------
    if cfgs.RUN.ckpt_dir is None:
        cfgs.RUN.ckpt_dir = ckpt.make_ckpt_dir(cfgs.RUN.ckpt_dir, run_name)
    else:
        step, ada_p, best_step, best_fid, best_ckpt_path, writer =\
            ckpt.load_StudioGAN_ckpts(ckpt_dir=cfgs.RUN.ckpt_dir,
                                      load_best=cfgs.RUN.load_best,
                                      Gen=Gen,
                                      Dis=Dis,
                                      g_optimizer=cfgs.OPTIMIZATION.g_optimizer,
                                      d_optimizer=cfgs.OPTIMIZATION.d_optimizer,
                                      run_name=run_name,
                                      apply_g_ema=cfgs.MODEL.apply_g_ema,
                                      Gen_ema=Gen_ema,
                                      ema=ema,
                                      is_train=cfgs.RUN.train,
                                      RUN=cfgs.RUN,
                                      logger=logger,
                                      global_rank=global_rank,
                                      device=local_rank)

    # -----------------------------------------------------------------------------
    # prepare parallel training
    # -----------------------------------------------------------------------------
    Gen, Dis, Gen_ema = model.prepare_parallel_training(Gen=Gen,
                                                        Dis=Dis,
                                                        Gen_ema=Gen_ema,
                                                        world_size=cfgs.OPTIMIZATION.world_size,
                                                        distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                                        synchronized_bn=cfgs.RUN.synchronized_bn,
                                                        apply_g_ema=cfgs.MODEL.apply_g_ema,
                                                        device=local_rank)

    # -----------------------------------------------------------------------------
    # load a pre-trained network (InceptionV3 or ResNet50 trained using SwAV)
    # -----------------------------------------------------------------------------
    if cfgs.RUN.eval:
        eval_model = pp.LoadEvalModel(eval_backbone=cfgs.RUN.eval_backbone,
                                      world_size=cfgs.OPTIMIZATION.world_size,
                                      distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                      device=local_rank)

        mu, sigma = pp.prepare_moments_calculate_ins(data_loader=eval_dataloader,
                                                     eval_model=eval_model,
                                                     splits=1,
                                                     cfgs=cfgs,
                                                     logger=logger,
                                                     device=local_rank)

    # -----------------------------------------------------------------------------
    # initialize WORKER for training and evaluating GAN
    # -----------------------------------------------------------------------------
    worker = WORKER(
        cfgs=cfgs,
        run_name=run_name,
        Gen=Gen,
        Dis=Dis,
        Gen_ema=Gen_ema,
        ema=ema,
        eval_model=eval_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        global_rank=global_rank,
        local_rank=local_rank,
        mu=mu,
        sigma=sigma,
        logger=logger,
        writer=writer,
        ada_p=ada_p,
        best_step=best_step,
        best_fid=best_fid,
        best_ckpt_path=best_ckpt_path,
    )

    # -----------------------------------------------------------------------------
    # train GAN until "total_setps" generator updates
    # -----------------------------------------------------------------------------
    if cfgs.RUN.train:
        if global_rank == 0: logger.info("Start training!")
        worker.training = True
        while step <= cfgs.OPTIMIZATION.total_steps:
            step = worker.train(current_step=step)

            if step % cfgs.RUN.save_every == 0:
                # visuailize fake images
                worker.visualize_fake_images(ncol=ncol)

                # evaluate GAN for monitoring purpose
                if cfgs.RUN.eval:
                    is_best = worker.evaluate(step=step)

                # save GAN in "./checkpoints/RUN_NAME/*"
                if global_rank == 0:
                    worker.save(step=step, is_best=is_best)

                # stop processes until all processes arrive
                if cfgs.RUN.distributed_data_parallel:
                    dist.barrier(worker.group)

    # -----------------------------------------------------------------------------
    # re-evaluate the best GAN and conduct ordered analyses
    # -----------------------------------------------------------------------------
    if global_rank == 0: logger.info("\n" + "-"*80)
    worker.training = False
    worker.standing_statistics = cfgs.RUN.standing_statistics
    worker.standing_max_batch = cfgs.RUN.standing_max_batch
    worker.standing_step = cfgs.RUN.standing_step

    best_step = ckpt.load_best_model(ckpt_dir=cfgs.RUN.ckpt_dir,
                                     Gen=Gen,
                                     Dis=Dis,
                                     apply_g_ema=cfgs.MODEL.apply_g_ema,
                                     Gen_ema=Gen_ema,
                                     ema=ema)

    if cfgs.RUN.eval:
        _ = worker.evaluate(step=best_step, writing=False)

    if cfgs.RUN.save_fake_images:
        worker.save_fake_images(png=True, npz=True)

    if cfgs.RUN.vis_fake_images:
        worker.visualize_fake_images(ncol=ncol)

    if cfgs.RUN.k_nearest_neighbor:
        worker.run_k_nearest_neighbor(dataset=eval_dataset, nrow=nrow, ncol=ncol)

    if cfgs.RUN.interpolation:
        worker.run_linear_interpolation(nrow=nrow,
                                        ncol=ncol,
                                        fix_z=True,
                                        fix_y=False)

        worker.run_linear_interpolation(nrow=nrow,
                                        ncol=ncol,
                                        fix_z=False,
                                        fix_y=True)

    if cfgs.RUN.frequency_analysis:
        worker.run_frequency_analysis(dataloader=eval_dataloader)

    if cfgs.RUN.tsne_analysis:
        worker.run_tsne(dataloader=eval_dataloader)
