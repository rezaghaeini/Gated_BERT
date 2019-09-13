# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os, re
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pyencoder import (BertConfig, BertForSequenceClassification, BertForMultiSequenceClassification)

from pytorch_transformers import (WEIGHTS_NAME, BertTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from .utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = None

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

def create_logger(args, name, to_disk=False, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def l0_penalty(w, eps, temp, gamma_zeta_ratio):
    def clip(x):
        return torch.min(torch.max(x, torch.zeros_like(x).fill_(eps)), torch.ones_like(x).fill_(1. - eps))
    
    return clip(torch.sigmoid(w.view(-1) - temp * gamma_zeta_ratio)).sum()

def sparsity_rate(mask, device):
    soft_is_zero = torch.eq(mask, 0.0).to(dtype=torch.float32)
    hard_is_zero = Variable(torch.le(mask, 0.5).to(dtype=torch.float32, device=device), requires_grad=False)
    return 100 * soft_is_zero.mean(), 100 * hard_is_zero.mean()

def smart_mask_regulaziation(w, strategy, loss_w, device, eps, temp, gamma_zeta_ratio):
    norm = None
    if strategy == 'L0':
        norm = l0_penalty(w, eps, temp, gamma_zeta_ratio)
    elif strategy == 'L1':
        norm =  torch.norm(w, p=1)
    elif strategy == 'L2':
        norm =  torch.norm(w, p=2)
    else:
        raise RuntimeError('Unsupported smart mask regualization: %s'%strategy)

    reg = F.l1_loss(norm, Variable(torch.tensor(0., device=device), requires_grad=False))
    return Variable(torch.tensor(loss_w, device=device), requires_grad=False) * reg

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir+'/runs/tensorboard')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight', 'head_weights', 'layer_weights']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_methods = {}
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.smart_head and args.regularize_head:
                tmp_model = model.module if hasattr(model, 'module') else model
                loss = loss + smart_mask_regulaziation(tmp_model.bert.head_weights, args.shm_reg_type,
                                                        args.shm_reg, args.device, tmp_model.bert.eps,
                                                        tmp_model.bert.temp, tmp_model.bert.gamma_zeta_ratio)
            if args.smart_pooling and args.regularize_pooling:
                tmp_model = model.module if hasattr(model, 'module') else model
                loss = loss + smart_mask_regulaziation(tmp_model.bert.layer_weights, args.lp_reg_type,
                                                        args.lp_reg, args.device, tmp_model.bert.eps,
                                                        tmp_model.bert.temp, tmp_model.bert.gamma_zeta_ratio)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.smart_head:
                    tmp_model = model.module if hasattr(model, 'module') else model
                    mask = tmp_model.bert.gate_mask(tmp_model.bert.head_weights, args.shm_reg_type,
                        tmp_model.bert.head_mask_size, args.device)
                    exact_sparsity, relaxed_sparsity = sparsity_rate(mask, args.device)
                    tb_writer.add_scalar('exact_sparsity', exact_sparsity, global_step)
                    tb_writer.add_scalar('relaxed_sparsity', relaxed_sparsity, global_step)
                    tb_writer.add_histogram('head_mask_weight', tmp_model.bert.head_weights, global_step)
                if args.smart_pooling:
                    tmp_model = model.module if hasattr(model, 'module') else model
                    mask = tmp_model.bert.gate_mask(tmp_model.bert.layer_weights, args.lp_reg_type,
                        tmp_model.bert.layer_mask_size, args.device)
                    for i in range(tmp_model.bert.config.num_hidden_layers):
                        tb_writer.add_scalar('layer_weight_%d'%i, mask[i], global_step)

                if args.local_rank in [-1, 0] and ((args.logging_steps > 0 and global_step % args.logging_steps == 0) or (args.save_steps > 0 and global_step % args.save_steps == 0)):
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            if key in best_methods:
                                if value > best_methods[key]:
                                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-{}'.format(str(key)))
                                    if not os.path.exists(output_dir):
                                        os.makedirs(output_dir)
                                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                                    model_to_save.save_pretrained(output_dir)
                                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                                    logger.info("Saving model checkpoint to %s", output_dir)
                                    best_methods[key] = value
                            else:
                                output_dir = os.path.join(args.output_dir, 'checkpoint-best-{}'.format(str(key)))
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                                logger.info("Saving model checkpoint to %s", output_dir)
                                best_methods[key] = value
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.local_rank in [-1, 0]:  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model, tokenizer)
            for key, value in results.items():
                if key in best_methods:
                    if value > best_methods[key]:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-{}'.format(str(key)))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        best_methods[key] = value
                else:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-{}'.format(str(key)))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '/' + args.task_name + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def evaluate_test(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_test_examples(args, eval_task, tokenizer)
        processor = processors[eval_task]()
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        output_eval_file = os.path.join(eval_output_dir, "test_%s_results.tsv"%prefix)
        with open(output_eval_file, "w") as writer:
            counter=0
            writer.write("index\tprediction\n")
            lbl_set = processor.get_labels()
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                              'labels':         None}
                    outputs = model(**inputs)
                    logits = outputs[0]
                    preds = logits.detach().cpu().numpy()
                    if args.output_mode == "classification":
                        preds = np.argmax(preds, axis=1)
                        for idx in range(preds.shape[0]):
                            writer.write("%d\t%s\n"%(counter, lbl_set[int(preds[idx])]))
                            counter += 1
                    elif args.output_mode == "regression":
                        preds = np.squeeze(preds)
                        for idx in range(preds.shape[0]):
                            writer.write("%d\t%s\n"%(counter, preds[idx]))
                            counter += 1

    return None

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def load_and_cache_test_examples(args, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'test', list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length), str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def main(args):
    global logger
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    if args.do_train:
        logger = create_logger(args, __name__, to_disk=True, log_file=args.output_dir+'/log.log')
    else:
        logger = create_logger(args, __name__, to_disk=True, log_file=args.output_dir+'/eval_log.log')

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print("--------------------------------")
        print("device: %s"%str(device))
        print("ngpu: %s"%str(args.n_gpu))
        print("--------------------------------")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if (not args.do_train) and (not os.path.exists(args.model_name_or_path + '/pytorch_model.bin')):
        args.model_name_or_path = args.model_name_or_path + '/checkpoint-final/'
    args.model_name_or_path = re.sub(r"/+", r"/", args.model_name_or_path)
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    if args.do_train:
        config.smart_head = args.smart_head
        config.smart_pooling = args.smart_pooling
        config.freeze_encoder = args.freeze_encoder
        config.shm_reg = args.shm_reg
        config.L0_start_mask = args.L0_start_mask
        config.hard_L0 = args.hard_L0
        config.noiseless_L0 = args.noiseless_L0
        config.shm_reg_type = args.shm_reg_type
        config.lp_reg = args.lp_reg
        config.L0_start_pooling = args.L0_start_pooling
        config.shm_init_method = args.shm_init_method
        config.lp_init_method = args.lp_init_method
        config.lp_reg_type = args.lp_reg_type
        config.gate_dropout = args.gate_dropout
        config.gate_normalize = args.gate_normalize
    tokenizer_address = args.model_name_or_path[:-1] if args.model_name_or_path[-1]=='/' else args.model_name_or_path
    if tokenizer_address.split('/')[-1][:10]=='checkpoint' and tokenizer_address.split('/')[-1]!="checkpoint-final":
        tokenizer_address = '/'.join(tokenizer_address.split('/')[:-1])
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else tokenizer_address, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    if args.prune_layer_count > 0:
        mask_values = []
        for i in range(config.num_hidden_layers):
            if config.num_hidden_layers-i <= args.prune_layer_count:
                mask_values.append(0.0)
            else:
                mask_values.append(1.0)
        model.bert._prune_layers(mask_values)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        output_dir = os.path.join(args.output_dir, 'checkpoint-final')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(output_dir)
        tokenizer = tokenizer_class.from_pretrained(output_dir)
        model.to(args.device)

    # Evaluation
    if args.local_rank in [-1, 0]:
        output_dir = os.path.join(args.output_dir, 'checkpoint-final')
        if not os.path.exists(output_dir):
            output_dir = args.output_dir
        checkpoints = [output_dir]
        checkpoints += glob.glob(args.output_dir + '/checkpoint-best*')
        for checkpoint in checkpoints:
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate_test(args, model, tokenizer, prefix=(checkpoint.split('-')[-1] if checkpoint!=args.output_dir else "final"))
            evaluate(args, model, tokenizer, prefix=(checkpoint.split('-')[-1] if checkpoint!=args.output_dir else "final"))

    return None
    # return results
