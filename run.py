# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from supar.config import Config
from supar.utils.logging import init_logger, logger
from supar.utils.parallel import get_device_count, get_free_port

from gec import Seq2SeqParser


def init(parser):
    parser.add_argument(
        '--ref',
        choices=['ss', 'mlr', 'mr'],
        default='ss',
        help='the way to process Multi-reference data'
        'duplicate input to num(refs) and split to multiple single-reference sample,'
        'select the ref with the most levenshtein ratio from input, keep Multi-refs for one input',
    )
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--conf', '-c', default='', help='path to config file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='num of threads')
    parser.add_argument('--workers', '-w', default=0, type=int, help='num of processes used for data loading')
    parser.add_argument('--cache', action='store_true', help='cache the data for fast loading')
    parser.add_argument('--binarize', action='store_true', help='binarize the data first')
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision for parsing')
    parser.add_argument('--dist', choices=['ddp', 'fsdp'], default='ddp', help='distributed training types')
    parser.add_argument('--wandb', action='store_true', help='wandb for tracking experiments')
    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args(unknown, args)
    args = Config.load(**vars(args), unknown=unknown)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if get_device_count() > 1:
        os.environ['MASTER_ADDR'] = 'tcp://localhost'
        os.environ['MASTER_PORT'] = get_free_port()
        mp.spawn(parse, args=(args,), nprocs=get_device_count())
    else:
        parse(0 if torch.cuda.is_available() else -1, args)


def parse(local_rank, args):
    Parser = args.pop('Parser')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if get_device_count() > 1:
        dist.init_process_group(
            backend='nccl', init_method=f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", world_size=get_device_count(), rank=local_rank
        )
    torch.cuda.set_device(local_rank)
    # init logger after dist has been initialized
    assert args.path.endswith(".pt")
    init_logger(logger, f"{args.path[:-3]}.{args.mode}.log", 'a' if args.get('checkpoint') else 'w')
    logger.info('\n' + str(args))

    args.local_rank = local_rank
    os.environ['RANK'] = os.environ['LOCAL_RANK'] = f'{local_rank}'
    if args.mode == 'train':
        parser = Parser.load(**args) if args.checkpoint else Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'predict':
        parser = Parser.load(**args)
        parser.predict(**args)
        parser.postprocess(**args)
        if args.scorer is not None:
            args.update({'eval': os.path.splitext(args.pred)[0] + f".{args.scorer.lower()}"})
            logger.info(f'Making evaluations on the data with {args.scorer}')
            parser.evaluate(**args)
            logger.info(f'Saving evaluated results to {args.eval}')


def main():
    parser = argparse.ArgumentParser(description='Create Seq2Seq GEC Parser.')
    parser.set_defaults(Parser=Seq2SeqParser)
    parser.add_argument('--eval-tgt', action='store_true', help='whether to evaluate tgt')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/clang8.train', help='path to train file')
    subparser.add_argument('--dev', default='data/bea19.dev', help='path to dev file')
    subparser.add_argument('--test', default=None, help='path to test file')
    subparser.add_argument('--vocab', default=tuple(), nargs='*', help='files for training vocabs')
    subparser.add_argument('--bin', default=None, help='dir to prerpocessed bin data')
    subparser.add_argument('--total_steps', default=None, type=int, help='total trining steps')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/nasgec_media.test', help='path to dataset')
    subparser.add_argument('--pred', default='pred.txt', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--select_min_loss', action='store_true', help='whether to select min loss reference from data')
    subparser.add_argument('--bin', default=None, help='dir to prerpocessed bin data')
    subparser.add_argument('--scorer', default=None, choices=["ChERRANT"], help='scorer for evaluation')
    if 'ChERRANT' in sys.argv:
        subparser.add_argument('--bpe', action='store_true', help='use bpe to split english words')
    subparser.add_argument('--gold', default='data/nasgec_media.test.m2', help='path to gold m2 file')
    init(parser)


if __name__ == "__main__":
    main()
