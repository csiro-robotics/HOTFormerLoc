# Adapted from MinkLoc3Dv2
# by Ethan Griffiths (Data61, Pullenvale)
# Train MinkLoc model

import os
import numpy as np
import torch
import tqdm
import pathlib
import wandb
from timm.utils.model_ema import ModelEmaV3
from timm.optim.lamb import Lamb

from misc.utils import TrainingParams, get_datetime, set_seed, update_params_from_dict
from models.losses.loss import make_losses
from models.model_factory import model_factory
from datasets.dataset_utils import make_dataloaders
from eval.pnv_evaluate import evaluate, print_eval_stats, pnv_write_eval_stats


def kdloss(y, teacher_scores):
    """
    Adapted from FasterViT repo:
    https://github.com/NVlabs/FasterViT/blob/main/fastervit/train.py#L356
    """
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()
    T = 3
    p = torch.nn.functional.log_softmax(y/T, dim=1)
    q = torch.nn.functional.softmax(teacher_scores/T, dim=1)
    l_kl = 50.0*kl_loss(p, q)
    return l_kl


def warmup(epoch: int):
    # Linear scaling lr warmup
    min_factor = 1e-3
    return max(float(epoch / num_warmup_epochs), min_factor)


def print_global_stats(phase, stats):
    s = f"{phase}  loss: {stats['loss']:.4f}   embedding norm: {stats['avg_embedding_norm']:.3f}  "
    if 'num_triplets' in stats:
        s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  " \
             f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
    if 'positives_per_query' in stats:
        s += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if 'best_positive_ranking' in stats:
        s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if 'recall' in stats:
        s += f"Recall@1: {stats['recall'][1]:.4f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "

    print(s, flush=True)


def print_stats(phase, stats):
    print_global_stats(phase, stats['global'])


def log_eval_stats(stats):
    eval_stats = {}
    for database_name in stats:
        eval_stats[database_name] = {}
        eval_stats[database_name]['recall@1%'] = stats[database_name]['ave_one_percent_recall']
        eval_stats[database_name]['recall@1'] = stats[database_name]['ave_recall'][0]
        eval_stats[database_name]['MRR'] = stats[database_name]['ave_mrr']
    return eval_stats


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def training_step(global_iter, model, phase, device, optimizer, loss_fn, mesa=0.0, model_ema=None):
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask = next(global_iter)
    batch = {e: batch[e].to(device, non_blocking=True) for e in batch}

    if phase == 'train':
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == 'train'):
        y = model(batch)
        stats = model.stats.copy() if hasattr(model, 'stats') else {}

        embeddings = y['global']

        loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
        temp_stats = tensors_to_numbers(temp_stats)
        stats.update(temp_stats)
        if phase == 'train':
            # Compute MESA loss
            if mesa > 0.0:
                with torch.no_grad():
                    ema_output = model_ema.module(batch)['global'].detach()
                kd = kdloss(embeddings, ema_output)
                loss += mesa * kd
                
            loss.backward()
            optimizer.step()
            
            if model_ema is not None:
                model_ema.update(model)

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
    return stats


def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn, mesa=0.0, model_ema=None):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

    assert phase in ['train', 'val']
    batch, positives_mask, negatives_mask = next(global_iter)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []
    embeddings_ema_l = []
    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device, non_blocking=True) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['global'])            
            # Compute MESA embeddings
            if mesa > 0.0:
                ema_output = model_ema.module(minibatch)['global'].detach()
                embeddings_ema_l.append(ema_output)

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_l, dim=0)
    if mesa > 0.0:
        embeddings_ema = torch.cat(embeddings_ema_l, dim=0)

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
        loss, stats = loss_fn(embeddings, positives_mask, negatives_mask)
        stats = tensors_to_numbers(stats)
        # Compute MESA loss
        if mesa > 0.0:
            kd = kdloss(embeddings, embeddings_ema)
            loss += mesa * kd
        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad

    # Delete intermediary values
    embeddings_l, embeddings, embeddings_ema_l, embeddings_ema, y, loss = [None]*6

    # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
    # network parameters using cached gradient of the loss w.r.t embeddings
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device, non_blocking=True) for e in minibatch}
                y = model(minibatch)
                embeddings = y['global']
                minibatch_size = len(embeddings)
                # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                # By default gradients are accumulated
                embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                i += minibatch_size

            optimizer.step()
            
            if model_ema is not None:
                model_ema.update(model)

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
    return stats


def do_train(params: TrainingParams = None, *args, **kwargs):
    # Set params for hyperparam search
    if params.hyperparam_search:
        if len(args) == 1 and isinstance(args[0], dict):  # This is required for submitit job arrays currently
            kwargs = args[0]
        assert kwargs != {}, 'No valid hyperparams were provided for search'
        params = update_params_from_dict(params, kwargs)
    params.print()
    # Seed RNG
    set_seed()
    # Create model class
    s = get_datetime()
    model = model_factory(params.model_params)
    model_name = params.model_params.model + '_' + s
    # Add SLURM job ID to prevent overwriting paths for jobs running at same time
    if 'SLURM_JOB_ID' in os.environ:
        model_name += f"_job{os.environ['SLURM_JOB_ID']}"
    weights_path = create_weights_folder(params.dataset_name)
    model_pathname = os.path.join(weights_path, model_name)
    print('Model name: {}'.format(model_name))
    
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    print('Model device: {}'.format(device))

    # Setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if params.mesa > 0.0:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV3(model, decay=0.9998)
    
    # set up dataloaders
    dataloaders = make_dataloaders(params, validation=params.validation)

    loss_fn = make_losses(params)

    # Training elements
    if params.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif params.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    elif params.optimizer == 'Lamb':
        optimizer_fn = Lamb
    else:
        raise NotImplementedError(f"Unsupported optimizer: {params.optimizer}")

    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr)
    else:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=params.gamma)
        elif params.scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    if params.warmup_epochs is not None:
        global num_warmup_epochs
        num_warmup_epochs = params.warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, scheduler], [params.warmup_epochs])

    if params.batch_split_size is None or params.batch_split_size == 0:
        train_step_fn = training_step
    else:
        # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
        train_step_fn = multistaged_training_step

    ###########################################################################
    # Initialize Weights&Biases logging service
    ###########################################################################

    params_dict = {e: params.__dict__[e] for e in params.__dict__ if e != 'model_params'}
    model_params_dict = {"model_params." + e: params.model_params.__dict__[e] for e in params.model_params.__dict__}
    params_dict.update(model_params_dict)
    n_params = sum([param.nelement() for param in model.parameters()])
    params_dict['num_params'] = n_params
    if params.wandb and not params.debug:
        wandb.init(project='HOTFormerLoc', config=params_dict)

    ###########################################################################
    #
    ###########################################################################

    # Training statistics
    stats = {'train': [], 'eval': []}
    best_avg_AR_1 = 0.0

    if 'val' in dataloaders:
        # Validation phase
        phases = ['train', 'val']
        stats['val'] = []
    else:
        phases = ['train']

    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        metrics = {'train': {}, 'val': {}, 'test': {}}      # Metrics for wandb reporting
        if epoch / params.epochs > params.mesa_start_ratio:
            mesa = params.mesa
        else:
            mesa = 0.0
        
        for phase in phases:
            running_stats = []  # running stats for the current epoch and phase
            count_batches = 0

            if phase == 'train':
                global_iter = iter(dataloaders['train'])
            else:
                global_iter = None if dataloaders['val'] is None else iter(dataloaders['val'])

            while True:
                count_batches += 1
                batch_stats = {}
                if params.debug and count_batches > 2:
                    break

                try:
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn, mesa, model_ema)
                    batch_stats['global'] = temp_stats

                except StopIteration:
                    # Terminate the epoch when one of dataloders is exhausted
                    break

                running_stats.append(batch_stats)

            # Compute mean stats for the phase
            epoch_stats = {}
            for substep in running_stats[0]:
                epoch_stats[substep] = {}
                for key in running_stats[0][substep]:
                    temp = [e[substep][key] for e in running_stats]
                    if type(temp[0]) is dict:
                        epoch_stats[substep][key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                    elif type(temp[0]) is np.ndarray:
                        # Mean value per vector element
                        epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                    else:
                        epoch_stats[substep][key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(phase, epoch_stats)

            # Log metrics for wandb
            metrics[phase]['loss1'] = epoch_stats['global']['loss']
            if 'num_non_zero_triplets' in epoch_stats['global']:
                metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']

            if 'positive_ranking' in epoch_stats['global']:
                metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']

            if 'recall' in epoch_stats['global']:
                metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]

            if 'ap' in epoch_stats['global']:
                metrics[phase]['AP'] = epoch_stats['global']['ap']

        # ******* FINALIZE THE EPOCH *******
        if scheduler is not None:
            scheduler.step()
        
        if not params.debug:
            if params.save_freq > 0 and epoch % params.save_freq == 0:
                epoch_pathname = f"{model_pathname}_e{epoch}.pth"
                print(f"Saving weights: {epoch_pathname}")
                torch.save(model.state_dict(), epoch_pathname)

        if params.eval_freq > 0 and epoch % params.eval_freq == 0:
            eval_stats = evaluate(model, device, params, log=False)
            print_eval_stats(eval_stats)
            metrics['test'] = log_eval_stats(eval_stats)
            # store best AR@1 on all test sets
            avg_AR_1 = metrics['test']['average']['recall@1']
            if avg_AR_1 > best_avg_AR_1:
                print(f"New best avg AR@1 at Epoch {epoch}: {best_avg_AR_1:.2f} -> {avg_AR_1:.2f}")
                best_avg_AR_1 = avg_AR_1
                if not params.debug:
                    best_model_pathname = f"{model_pathname}_best.pth"
                    print(f"Saving weights: {best_model_pathname}")
                    torch.save(model.state_dict(), best_model_pathname)

        if params.wandb and not params.debug:
            wandb.log(metrics)

        if params.batch_expansion_th is not None:
            # Dynamic batch size expansion based on number of non-zero triplets
            # Ratio of non-zero triplets
            le_train_stats = stats['train'][-1]  # Last epoch training stats
            rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
            if rnz < params.batch_expansion_th:
                dataloaders['train'].batch_sampler.expand_batch()

    print('')

    # Save final model weights
    if not params.debug:
        final_model_path = model_pathname + '_final.pth'
        print(f"Saving weights: {final_model_path}")
        torch.save(model.state_dict(), final_model_path)

    # Evaluate the final
    # PointNetVLAD datasets evaluation protocol
    stats = evaluate(model, device, params, log=False)
    print_eval_stats(stats)

    print('.')

    # Append key experimental metrics to experiment summary file
    if not params.debug:
        model_params_name = os.path.split(params.model_params.model_params_path)[1]
        config_name = os.path.split(params.params_path)[1]
        model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]
        prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)

        pnv_write_eval_stats(f"pnv_{params.dataset_name}_results.txt", prefix, stats)

    # Return optimization value (to minimize)
    return (1 - best_avg_AR_1/100.0)

def create_weights_folder(dataset_name : str):
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights', dataset_name)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
