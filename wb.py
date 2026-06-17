"""wandb: default-on, offline (compute nodes have no internet -> `wandb sync` later),
never blocks training. Returns report_to for TrainingArguments."""

import os


def wandb_init(args, stage):
    if int(os.environ.get("RANK", "0")) != 0:  # rank0 only under DDP
        return []
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb.init(project="codi_trace", name=f"{stage}-{os.path.basename(args.output_dir)}",
                   dir=args.output_dir, config=vars(args))
        return ["wandb"]
    except Exception as e:
        print(f"wandb disabled: {e}")
        return []
