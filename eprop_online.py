'''
per-timestep (online) training step for the e-prop models.

The batched step in nested_train.py / main.py runs the whole sequence, then calls
backward once. For the e-prop models that pins every timestep's eligibility
traces in memory simultaneously (~830MB at recurrent_dims=[32, 64, 32],
batch=64, T=300) -- test_eprop_streaming_equivalence.py measures it.

online_train_step instead backwards each timestep's loss term the moment its
logits exist, which frees that timestep's traces immediately. The accumulated
gradient is mathematically identical to the batched one; test_eprop_streaming_
equivalence.py asserts it.

Shared by nested_train_eprop.py and main_eprop.py so the 1/T_supervised scaling
below lives in exactly one place -- get it wrong and the gradient is silently
off by a factor of T.
'''

import sys

from utils.encoding import unpack_batch
from utils.readout import prepare_loss_inputs


def online_train_step(model, train_loader, encoder, optimizer, loss_fn, device,
                      beat_bounds=None, binary=True, epoch=None, log_every=None):
    '''
    one epoch of per-timestep e-prop training.

    Same signature and return value as nested_train.train_step, plus optional
    epoch/log_every for main_eprop's progress printing.

    optimizer.step() still runs once per batch, not once per timestep: .grad is
    accumulated across the sequence exactly as the batched path accumulates it
    during its single backward. Stepping per timestep would be a different
    learning algorithm and would not reproduce existing results.
    '''

    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        data, target, stats = unpack_batch(batch)

        data = encoder.encode(data, *stats)
        data, target = data.to(device), target.to(device)

        # supervise only the labeled beat span -- the pre/post context is fed to
        # the model as input but carries the wrong label. Timesteps outside the
        # span still need their forward pass (they carry state and traces
        # forward), they just contribute no loss term.
        start, end = beat_bounds if beat_bounds is not None else (0, data.shape[1])
        num_supervised = end - start

        # accumulated in a one-element list rather than a `nonlocal` so the
        # closure stays readable
        batch_loss = [0.0]

        def on_step(t, output):
            if not start <= t < end:
                # nothing retains `output`, so timestep t's graph dies here
                return

            # reuse prepare_loss_inputs with a singleton time axis so the binary
            # and multiclass shaping cannot drift from the batched path
            step_output, step_target = prepare_loss_inputs(
                output.unsqueeze(1), target[:, t:t + 1], binary)

            # loss_fn reduces over the batch only. The batched path takes one
            # mean over (batch x num_supervised), so each step is scaled by
            # 1/num_supervised for the sum over t to match it exactly.
            loss = loss_fn(step_output, step_target) / num_supervised
            loss.backward()

            batch_loss[0] += loss.item()

        model.stream_forward(data, on_step)

        optimizer.step()

        if log_every is not None and (batch_idx + 1) % log_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), batch_loss[0]))
            sys.stdout.flush()

        epoch_loss += batch_loss[0] * data.shape[0]

    epoch_loss /= len(train_loader.dataset)

    return epoch_loss
