# -*- coding: utf-8 -*-

import torch
from datasets import load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer)

import fla  # noqa
from flame.data import DataCollatorForLanguageModeling
from flame.logging import LogCallback, get_logger
from flame.parser import get_train_args

logger = get_logger(__name__)


def main():
    args = get_train_args()
    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))
    if args.from_config:
        logger.info("All model params are randomly initialized for from-scratch training.")
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model_name_or_path))
    else:
        logger.info(f"Loading pretrained checkpoint {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.train()

    if getattr(model.config, "cut_cross_entropy", False):
        try:
            from cut_cross_entropy import linear_cross_entropy
        except ImportError as e:
            raise ImportError(f"You use `cut_cross_entropy` but you did not have CCE. Install it first!: {e}")

    trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
    logger.info(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
    logger.info(f"{tokenizer}\n{model}\n{model.config}")

    logger.info(f"Loading the `{args.split}` split directly from the cache {args.cache_dir}...")
    dataset = load_from_disk(args.cache_dir)
    logger.info(f"{dataset}")
    logger.info(f"Shuffling the dataset with seed {args.seed}")
    dataset = dataset.shuffle(seed=args.seed)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    if args.lr_scheduler_type == 'warmup_stable_decay':
        args.lr_scheduler_kwargs = {
            'num_stable_steps': args.max_steps * 0.9 - args.warmup_steps,
            'num_decay_steps': args.max_steps * 0.1
        }


    # model = torch.compile(model)

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        train_dataset=dataset
    )

    if args.do_profiling:
        from edd_utils import ProfCallback

        profiling_output_dir = f"profiler_output_{args.output_dir}"

        callback = ProfCallback(
            cpu=True,
            cuda=True,
            output_dir=profiling_output_dir
        )

        new_max_steps = (callback.active_steps + callback.warmup_steps + callback.wait_steps) * (callback.repeat + 1) + 2
        print("Do profiling is activated, therefore we should reduce the amount of steps ", 
              f"to {new_max_steps}")
        print(f"We will save the profiling outputs to {profiling_output_dir}")

        args.max_steps = new_max_steps

        trainer.add_callback(callback)

    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    if not args.do_profiling:
        trainer.log_metrics("train", results.metrics)
        trainer.save_metrics("train", results.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
