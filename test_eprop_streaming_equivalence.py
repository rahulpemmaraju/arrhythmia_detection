'''
assert that per-timestep e-prop training matches the batched path.

Each network in models/eprop_hgrsnn.py carries a second copy of its time loop --
stream_forward -- that hands each timestep's logits to a callback so the loss can
be backwarded immediately, which is what makes peak memory O(1) in T instead of
O(T). That is only legitimate if the accumulated gradient is the same one the
batched forward + single backward produces. This checks:

  1. every parameter's .grad matches between the two paths
  2. the reported epoch loss matches
  3. some gradients are actually non-zero (otherwise 1 and 2 are vacuous)
  4. the streaming path really does retain less memory, and flat in T

Run from the package root:  python test_eprop_streaming_equivalence.py

The two time loops are separate implementations on purpose -- forward() stays
untouched as the reference. This test is what catches them drifting apart.
'''

import gc
import sys

import torch
import torch.nn as nn

from models import create_model
from utils.readout import prepare_loss_inputs

torch.set_num_threads(1)

# float32 through a few hundred timesteps of differently-ordered summation; the
# two paths are algebraically identical but not bitwise
RTOL = 1e-4
ATOL = 1e-6


def build(model_type, recurrent_dims, out_dims):
    configs = {
        'kwargs': {
            'model_type': model_type,
            'neuron_type': 'lif',
            'out_type': 'linear',
            'bias': False,
            'recurrent_dims': recurrent_dims,
            'out_dims': out_dims,
        },
        'x0': {
            'alpha_0': 0.9, 'threshold_0': 1.0,
            'alpha_1': 0.9, 'threshold_1': 1.0,
            'alpha_2': 0.9, 'threshold_2': 1.0,
            'lr': 0.001,
        },
    }
    return create_model(configs, 'x0')


def make_batch(B, T, out_dims, binary, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, 1, generator=g)

    if binary:
        # the label is constant across the window, as in the real dataset
        labels = torch.randint(0, 2, (B, 1), generator=g).float()
    else:
        labels = torch.randint(0, out_dims, (B, 1), generator=g).float()

    return x, labels.expand(B, T).contiguous()


def batched_grads(model, x, target, loss_fn, bounds, binary):
    '''the current path: one forward, one backward (nested_train.train_step)'''

    model.zero_grad(set_to_none=True)
    model.train()

    output, _ = model(x)

    start, end = bounds
    output, target = output[:, start:end], target[:, start:end]
    output, target = prepare_loss_inputs(output, target, binary)

    loss = loss_fn(output, target)
    loss.backward()

    return snapshot(model), loss.item()


def streaming_grads(model, x, target, loss_fn, bounds, binary):
    '''the new path: backward once per timestep (eprop_online.online_train_step)'''

    model.zero_grad(set_to_none=True)
    model.train()

    start, end = bounds
    num_supervised = end - start
    total = [0.0]

    def on_step(t, output):
        if not start <= t < end:
            return

        step_output, step_target = prepare_loss_inputs(
            output.unsqueeze(1), target[:, t:t + 1], binary)

        loss = loss_fn(step_output, step_target) / num_supervised
        loss.backward()
        total[0] += loss.item()

    model.stream_forward(x, on_step)

    return snapshot(model), total[0]


def snapshot(model):
    return {name: (p.grad.detach().clone() if p.grad is not None else None)
            for name, p in model.named_parameters()}


def live_tensor_mb():
    gc.collect()
    seen, total = set(), 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                storage = obj.untyped_storage()
                if id(storage) not in seen:
                    seen.add(id(storage))
                    total += storage.nbytes()
        except Exception:
            pass

    return total / 1e6


def compare(case, batched, streaming, loss_a, loss_b):
    ok = True
    worst_name, worst_rel = None, 0.0
    nonzero = 0

    names = sorted(set(batched) | set(streaming))

    for name in names:
        a, b = batched[name], streaming[name]

        if (a is None) != (b is None):
            print('  FAIL {}: grad is None in only one path ({} vs {})'.format(
                name, 'None' if a is None else 'set', 'None' if b is None else 'set'))
            ok = False
            continue

        if a is None:
            continue

        if a.abs().max().item() > 0:
            nonzero += 1

        if not torch.allclose(a, b, rtol=RTOL, atol=ATOL):
            print('  FAIL {}: max abs diff {:.3e}'.format(name, (a - b).abs().max().item()))
            ok = False

        denom = a.abs().max().item()
        if denom > 0:
            rel = (a - b).abs().max().item() / denom
            if rel > worst_rel:
                worst_rel, worst_name = rel, name

    if nonzero == 0:
        print('  FAIL: every gradient was zero -- the comparison proves nothing')
        ok = False

    loss_diff = abs(loss_a - loss_b)
    if loss_diff > 1e-5 * max(1.0, abs(loss_a)):
        print('  FAIL loss: batched {:.8f} vs streaming {:.8f}'.format(loss_a, loss_b))
        ok = False

    print('  {} {}: {}/{} params with nonzero grad, worst relative diff {:.2e}{}, '
          'loss {:.6f} vs {:.6f}'.format(
              'PASS' if ok else 'FAIL', case, nonzero,
              sum(1 for n in names if batched[n] is not None), worst_rel,
              ' ({})'.format(worst_name) if worst_name else '', loss_a, loss_b))

    return ok


def run_case(case, model_type, recurrent_dims, out_dims, binary, T, bounds, B=8):
    model = build(model_type, recurrent_dims, out_dims)
    x, target = make_batch(B, T, out_dims, binary)
    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()

    bounds = bounds if bounds is not None else (0, T)

    a, loss_a = batched_grads(model, x, target, loss_fn, bounds, binary)
    b, loss_b = streaming_grads(model, x, target, loss_fn, bounds, binary)

    return compare(case, a, b, loss_a, loss_b)


def run_memory_check():
    print('\nretained tensors during the backward pass (B=64, recurrent_dims=[32, 64, 32]):')
    print('  {:>6}  {:>14}  {:>14}'.format('T', 'batched', 'streaming'))

    loss_fn = nn.BCEWithLogitsLoss()
    rows = []

    for T in (50, 150, 300):
        model = build('four_layer_hgrsnn_eprop', [32, 64, 32], 1)
        x, target = make_batch(64, T, 1, binary=True)

        model.zero_grad(set_to_none=True)
        model.train()

        base = live_tensor_mb()
        output, _ = model(x)
        batched_peak = live_tensor_mb() - base

        out, tgt = prepare_loss_inputs(output, target, True)
        loss_fn(out, tgt).backward()
        del output, out

        # peak inside the streaming loop, sampled at the last supervised step
        model.zero_grad(set_to_none=True)
        peak = [0.0]
        base = live_tensor_mb()

        def on_step(t, o):
            step_out, step_tgt = prepare_loss_inputs(o.unsqueeze(1), target[:, t:t + 1], True)
            (loss_fn(step_out, step_tgt) / T).backward()
            if t == T - 1:
                peak[0] = live_tensor_mb() - base

        model.stream_forward(x, on_step)

        rows.append((T, batched_peak, peak[0]))
        print('  {:>6}  {:>11.1f} MB  {:>11.1f} MB'.format(T, batched_peak, peak[0]))

    # the point of the exercise: batched grows with T, streaming does not
    growth_batched = rows[-1][1] - rows[0][1]
    growth_streaming = abs(rows[-1][2] - rows[0][2])

    ok = growth_batched > 100 and growth_streaming < 5 and rows[-1][2] < rows[-1][1] / 10

    print('  {}: batched grew {:.1f} MB from T=50 to T=300, streaming moved {:.1f} MB'.format(
        'PASS' if ok else 'FAIL', growth_batched, growth_streaming))

    return ok


if __name__ == '__main__':
    print('gradient equivalence, batched vs per-timestep streaming:')

    results = [
        run_case('four_layer binary, full window',
                 'four_layer_hgrsnn_eprop', [32, 64, 32], 1, True, T=60, bounds=None),
        run_case('four_layer binary, beat_bounds subset',
                 'four_layer_hgrsnn_eprop', [32, 64, 32], 1, True, T=60, bounds=(12, 47)),
        run_case('four_layer multiclass, beat_bounds subset',
                 'four_layer_hgrsnn_eprop', [32, 64, 32], 4, False, T=60, bounds=(12, 47)),
        run_case('three_layer binary, beat_bounds subset',
                 'three_layer_hgrsnn_eprop', [16, 24, 1], 1, True, T=60, bounds=(12, 47)),
        run_case('two_layer binary, full window',
                 'two_layer_hgrsnn_eprop', [16, 1], 1, True, T=60, bounds=None),
    ]

    results.append(run_memory_check())

    print('\n{}: {}/{} checks passed'.format(
        'PASS' if all(results) else 'FAIL', sum(results), len(results)))

    sys.exit(0 if all(results) else 1)
