def center_bounds(seq_len, center_frac=1.0):
    '''
    start/end indices of the central center_frac of a length-seq_len sequence.

    create_beat_dataset centers each window on the beat it is labeled with, so
    the flanking timesteps hold *neighboring* beats carrying the center beat's
    label. supervising and reading out over the whole window therefore trains
    the model to call a normal neighbor abnormal whenever the center beat is,
    and dilutes the summed-logit score with those timesteps.

    restricting loss and readout to the center keeps the context in the input --
    the recurrent state still sees the previous RR interval -- without letting it
    corrupt the target. center_frac=1.0 is the full window (no masking).

    seq_len: number of timesteps in the window
    center_frac: fraction of the window to keep, centered on the beat
    '''

    if not 0 < center_frac <= 1.0:
        raise ValueError(f'center_frac must be in (0, 1], got {center_frac}')

    if center_frac == 1.0:
        return 0, seq_len

    width = max(1, int(round(seq_len * center_frac)))
    start = (seq_len - width) // 2

    return start, start + width
