"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


class WindowStartSampler:
    """Draw training windows either randomly or in shuffled coverage passes.

    A coverage pass uses a random offset, tiles the corpus with non-overlapping
    windows, and shuffles those windows. Leftover starts carry into the next
    batch so every optimizer step keeps the requested batch size.
    """
    def __init__(self, token_count, batch_size, context, mode, generator):
        if token_count <= context + 1:
            raise ValueError('Training text is shorter than one full window.')
        self.token_count = token_count
        self.batch_size = batch_size
        self.context = context
        self.mode = mode
        self.generator = generator
        self.pending = torch.empty(0, dtype=torch.long)
        self.coverage_passes = 0

    def _new_coverage_pass(self):
        offset = torch.randint(self.context, (1,), generator=self.generator).item()
        starts = torch.arange(offset, self.token_count-self.context, self.context)
        starts = starts[torch.randperm(len(starts), generator=self.generator)]
        self.coverage_passes += 1
        return starts

    def sample(self):
        if self.mode == 'random':
            return torch.randint(
                self.token_count-self.context-1,
                (self.batch_size,), generator=self.generator)
        while len(self.pending) < self.batch_size:
            self.pending = torch.cat((self.pending, self._new_coverage_pass()))
        starts, self.pending = self.pending[:self.batch_size], self.pending[self.batch_size:]
        return starts


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--resume', type=Path,
                   help='Continue from model weights in an existing checkpoint.')
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--sampling', choices=['random','coverage'], default='random',
                   help='random samples with replacement; coverage shuffles non-overlapping windows.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--train-last-blocks', type=int, default=0,
                   help='Freeze the model except for the final N Transformer blocks.')
    p.add_argument('--dropout', type=float,
                   help='Override dropout for a continuation run; tensor shapes are unchanged.')
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--save-best', action='store_true',
                   help='Save best-validation.pt at intermediate validation intervals.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        p.error('Steps, batch size, and learning rate must be positive.')
    if args.warmup_steps < 0:
        p.error('Warmup steps cannot be negative.')
    if args.train_last_blocks < 0:
        p.error('--train-last-blocks cannot be negative.')
    if args.dropout is not None and not 0 <= args.dropout < 1:
        p.error('--dropout must be in [0, 1).')
    if args.weight_decay < 0:
        p.error('--weight-decay cannot be negative.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    if args.dropout is not None:
        config['dropout'] = args.dropout
    model, implementation_sha = make_model(args.implementation, config, device)
    prior_train_tokens = 0
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=True)
        if resumed['protocol'] != PROTOCOL:
            p.error('Resume checkpoint belongs to a different protocol.')
        resume_config = dict(resumed['config'])
        if args.dropout is not None:
            resume_config['dropout'] = args.dropout
        if resumed['implementation'] != args.implementation or resume_config != config:
            p.error('Resume checkpoint implementation/config does not match this run.')
        model.load_state_dict(resumed['model'])
        prior_train_tokens = int(resumed.get('train_tokens', 0))
    if args.train_last_blocks:
        if not hasattr(model, 'blocks') or args.train_last_blocks > len(model.blocks):
            p.error('--train-last-blocks exceeds the model depth.')
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for block in model.blocks[-args.train_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=.001, weight_decay=args.weight_decay)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    start_sampler = WindowStartSampler(
        len(tokens), args.batch_size, model.context, args.sampling, rng)
    positions = torch.arange(model.context+1, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best_validation_bpb = float('inf')
    for step in range(args.steps):
        starts = start_sampler.sample().to(device)
        batch = tokens[starts[:,None]+positions]
        warmup = 1. if args.warmup_steps == 0 else min(1., (step+1)/args.warmup_steps)
        learning_rate = args.learning_rate * warmup * (
            .1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            if hasattr(model, 'training_loss'):
                loss = model.training_loss(batch[:, :-1], batch[:, 1:])
            else:
                loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters,1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if args.save_best and intermediate['bpb'] < best_validation_bpb:
                best_validation_bpb = intermediate['bpb']
                torch.save({
                    'protocol': PROTOCOL, 'implementation': args.implementation,
                    'config': config,
                    'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    'seed': args.seed,
                    'train_tokens': prior_train_tokens + (step+1)*args.batch_size*model.context,
                    'sampling': args.sampling, 'selected_step': step+1,
                    'validation_bpb': intermediate['bpb'],
                }, args.run_dir/'best-validation.pt')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    total_train_tokens = prior_train_tokens + args.steps*args.batch_size*model.context
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':total_train_tokens,
                'sampling':args.sampling},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':total_train_tokens,'run_train_tokens':args.steps*args.batch_size*model.context,
              'resume':str(args.resume) if args.resume else None,
              'learning_rate':args.learning_rate,'warmup_steps':args.warmup_steps,
              'weight_decay':args.weight_decay,'dropout_override':args.dropout,
              'sampling':args.sampling,
              'train_last_blocks':args.train_last_blocks,
              'coverage_passes':start_sampler.coverage_passes,
              'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
