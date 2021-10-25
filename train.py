import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import cuda, optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from config import get_args
from Dataset import RerankDataset_TopKSIM, collate_tuples
from network import RerankTransformer
from utils import MetricLogger, WarmupCos_Scheduler, create_optimizer, get_current_dir, init_distributed_mode, is_main_process


def main(args):
    init_distributed_mode(args)
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if args.distributed:
        ngpus_per_node = cuda.device_count()
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print('>> batch size per node:{}'.format(args.batch_size))
        print('>> num workers per node:{}'.format(args.num_workers))

    dataset_names = list(eval(args.dataset_names))
    output_dir = os.path.join(get_current_dir(), 'Experiment_{}_trainingInfo'.format(args.test_name))
    train_dataset = RerankDataset_TopKSIM(names=dataset_names, mode='train', topk=args.topk, sim_len=args.sim_len)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=False)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_tuples, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True, collate_fn=collate_tuples)

    model = RerankTransformer(embed_dim=args.embed_dim,
                              topk_dim=args.sim_len,
                              depth=args.depth,
                              num_heads=args.num_heads,
                              mlp_ratio=args.mlp_ratio,
                              qkv_bias=args.qkv_bias,
                              drop_rate=args.drop_rate,
                              attn_drop_rate=args.attn_drop_rate,
                              drop_path_rate=args.drop_path_rate).to(device)

    model_without_ddp = model

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('>> number of params:{:.2f}M'.format(n_parameters / 1e6))

    # define optimizer
    param_dicts = create_optimizer(args.weight_decay, model_without_ddp)
    optimizer = optim.SGD(param_dicts, lr=args.base_lr * args.batch_size / 256, weight_decay=args.weight_decay, momentum=args.momentum)

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    lr_scheduler = WarmupCos_Scheduler(optimizer=optimizer,
                                       warmup_epochs=args.warmup_epochs,
                                       warmup_lr=args.warmup_lr * args.batch_size * args.update_every / 256,
                                       num_epochs=args.num_epochs,
                                       base_lr=args.base_lr * args.batch_size * args.update_every / 256,
                                       final_lr=args.final_lr * args.batch_size * args.update_every / 256,
                                       iter_per_epoch=int(len(train_loader) / args.update_every))

    lr_scheduler.iter = max(int(len(train_loader) * start_epoch / args.update_every), 0)

    # Start training
    metric_logger = MetricLogger(delimiter=" ")
    print_freq = 10
    model_path = None
    scaler = GradScaler()
    Gradient_Logger = {'topk projecter': [], 'last transformer block': []}
    Loss_logger = {'Contrastive loss': [], 'MSE loss': []}

    for epoch in range(start_epoch, args.stop_at_epoch):
        header = '>> Train Epoch: [{}]'.format(epoch)
        optimizer.zero_grad()
        for idx, (sim_list, targets) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            model.train()
            targets = targets.to(device, non_blocking=True)
            exp_sim, MSE_loss = model(sim_list.to(device, non_blocking=True).half())
            Contrastive_loss = torch.mean(-1.0 * torch.log(torch.sum(exp_sim * targets, dim=-1) / torch.sum(exp_sim, dim=-1)))
            loss = Contrastive_loss + args.alpha * MSE_loss

            if not math.isfinite(Contrastive_loss.item()):
                print(">> Contrastive loss is nan, skip to the next iteration")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            if not math.isfinite(MSE_loss.item()):
                print(">> MSE loss is nan, skip to the next iteration")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            metric_logger.meters['Contrastive loss'].update(Contrastive_loss.item())
            metric_logger.meters['MSE loss'].update(MSE_loss.item())
            with torch.no_grad():
                topk_project_gradient_norm = max(p.grad.detach().abs().max().cpu() for p in model_without_ddp.topk_proj.parameters())
                transformer_gradient_norm = max(p.grad.detach().abs().max().cpu() for p in model_without_ddp.blocks[-1].parameters())
                metric_logger.meters['topk gradient'].update(topk_project_gradient_norm.item())
                metric_logger.meters['last layer gradient'].update(transformer_gradient_norm.item())

            if (idx + 1) % args.update_every == 0:
                if args.clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                lr = lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (idx + 1) % 50 == 0:
                if is_main_process():
                    Gradient_Logger['topk projecter'].append(metric_logger.meters['topk gradient'].avg)
                    Gradient_Logger['last transformer block'].append(metric_logger.meters['last layer gradient'].avg)
                    Loss_logger['Contrastive loss'].append(metric_logger.meters['Contrastive loss'].avg)
                    Loss_logger['MSE loss'].append(metric_logger.meters['MSE loss'].avg)
                    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
                    fig.tight_layout()
                    axes = axes.flatten()
                    for (key, value) in Gradient_Logger.items():
                        axes[0].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[0].legend(loc='upper center', shadow=True, fontsize='medium')
                    axes[0].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[0].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[0].set_xlabel('iter')
                    axes[0].set_ylabel("gradient,inf-norm")
                    axes[0].minorticks_on()
                    for (key, value) in Loss_logger.items():
                        axes[1].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[1].legend(loc='upper right', shadow=True, fontsize='medium')
                    axes[1].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[1].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[1].set_xlabel('iter')
                    axes[1].set_ylabel("loss")
                    axes[1].minorticks_on()
                    plt.savefig(os.path.join(output_dir, 'training_logger.png'))
                    plt.close()
        if is_main_process():
            # Save checkpoint
            model_path = os.path.join(output_dir, 'checkpoint-epoch{}.pth'.format(epoch + 1))
            torch.save({'epoch': epoch + 1, 'state_dict': model_without_ddp.state_dict()}, model_path)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
