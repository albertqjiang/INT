#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Modified by Albert Qiaochu Jiang for evaluating INT problems
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
from copy import deepcopy
import fileinput
import logging
import math
import sys
import time
import os
import pickle
import json

import numpy as np

import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

from proof_system.prover import Prover
from data_generation.seq_prefairseq import chared_to_string, filter_arrow
from proof_system.all_axioms import all_axioms_to_prove


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')


Batch = namedtuple('Batch', 'ids src_tokens src_lengths constraints')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if args.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [task.target_dictionary.encode_line(
                encode_fn_target(constraint),
                append_eos=False,
                add_if_not_exist=False,
            ) for constraint in constraint_list]

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    if args.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths, constraints=constraints_tensor),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch['id']
        src_tokens = batch['net_input']['src_tokens']
        src_lengths = batch['net_input']['src_lengths']
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


def main(args):
    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.constraints:
        logger.warning("NOTE: Constrained decoding currently assumes a shared subword vocabulary.")

    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Type the input sentence and press return:')
    start_id = 0

    ###################################################
    # INT stuff starts here
    ###################################################
    if not os.path.isdir(args.dump_path):
        os.makedirs(args.dump_path)
    f_out = open(args.results_path, "w")

    out_buffer = ""

    # Load test problems to evaluate
    test_problems = pickle.load(open(os.path.join(args.test_problems_path, "test_problems.pkl"), "rb"))
    proofs_closed = 0
    for test_problem in test_problems:
        for _ in range(5):
            # Every theorem gets three goes
            test_step_1 = test_problem[1]
            test_prover = Prover(axioms=all_axioms_to_prove,
                                 conditions=test_step_1["observation"]["ground_truth"],
                                 objectives=test_step_1["observation"]["objectives"],
                                 prove_direction="backward")
            for _ in range(15):
                if test_prover.is_proved():
                    break
                source = test_prover.parser.observation_to_source(test_prover.get_observation())
                inputs = [" ".join(filter_arrow(source))]

                ###################################################
                # INT stuff stops here
                ###################################################

                # for inputs in buffered_read(args.input, args.buffer_size):
                results = []
                for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                    bsz = batch.src_tokens.size(0)
                    src_tokens = batch.src_tokens
                    src_lengths = batch.src_lengths
                    constraints = batch.constraints
                    if use_cuda:
                        src_tokens = src_tokens.cuda()
                        src_lengths = src_lengths.cuda()
                        if constraints is not None:
                            constraints = constraints.cuda()

                    sample = {
                        'net_input': {
                            'src_tokens': src_tokens,
                            'src_lengths': src_lengths,
                        },
                    }
                    translate_start_time = time.time()
                    translations = task.inference_step(generator, models, sample, constraints=constraints)
                    translate_time = time.time() - translate_start_time
                    total_translate_time += translate_time
                    list_constraints = [[] for _ in range(bsz)]
                    if args.constraints:
                        list_constraints = [unpack_constraints(c) for c in constraints]
                    for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                        src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                        constraints = list_constraints[i]
                        results.append((start_id + id, src_tokens_i, hypos,
                                        { "constraints": constraints,
                                          "time": translate_time / len(translations) }
                                    ))

                # sort output to match input order
                for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                        out_buffer += 'S-{}\t{}\n'.format(id_, src_str)
                        out_buffer += "W-{}\t{:.3f}\tseconds\n".format(id_, info["time"])
                        for constraint in info["constraints"]:
                            out_buffer += "C-{}\t{}\n".format(id_, tgt_dict.string(constraint, args.remove_bpe))

                    # Process top predictions
                    for hypo in hypos[:min(len(hypos), args.nbest)]:
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'],
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                        )
                        detok_hypo_str = decode_fn(hypo_str)
                        score = hypo['score'] / math.log(2)  # convert to base 2
                        # original hypothesis (after tokenization and BPE)
                        out_buffer += 'H-{}\t{}\t{}\n'.format(id_, score, hypo_str)

                        exec_string = chared_to_string(hypo_str.split())
                        out_buffer += "Execution string:" + exec_string + "\n"
                        try:
                            result = test_prover.apply_theorem_seq_style(exec_string)
                            # out_buffer += result)
                        except Exception:
                            continue
                        if test_prover.is_proved():
                            proofs_closed += 1
                            break
                        out_buffer += "Objec: " + " ".join(filter_arrow(
                            test_prover.parser.observation_to_source(test_prover.get_observation()))) + "\n"

                        if result["progress"]:
                            break

                        # detokenized hypothesis
                        out_buffer += 'D-{}\t{}\t{}\n'.format(id_, score, detok_hypo_str)
                        out_buffer += 'P-{}\t{}\n'.format(
                            id_,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                # convert from base e to base 2
                                hypo['positional_scores'].div_(math.log(2)).tolist(),
                            ))
                        )
                        if args.print_alignment:
                            alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                            out_buffer += 'A-{}\t{}\n'.format(
                                id_,
                                alignment_str
                            )

                # update running id_ counter
                start_id += len(inputs)
            if test_prover.is_proved():
                out_buffer += "Proof closed\n"
            if not test_prover.is_proved():
                out_buffer += "Proof failed\n"

            if test_prover.is_proved():
                break

        if len(out_buffer) >= 10000:
            f_out.write(out_buffer)
            out_buffer = ""

    logger.info("Total time: {:.3f} seconds; translation time: {:.3f}\n".format(time.time() - start_time, total_translate_time))
    f_out.write("We closed {} proofs out of {} in total. Proportion: {}\n".format(
        proofs_closed, len(test_problems), proofs_closed/len(test_problems)))
    print("We closed {} proofs out of {} in total. Proportion: {}\n".format(
        proofs_closed, len(test_problems), proofs_closed/len(test_problems)))
    json.dump('{percent:.2%}'.format(percent=proofs_closed / len(test_problems)),
              open(os.path.join(args.dump_path, args.fingerprint), "w"))


def cli_main():
    parser = options.get_interactive_evaluation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == '__main__':
    cli_main()
