import numpy as np

def apply_packet_loss(wav, sr, loss_pct=0):
    if loss_pct <= 0:
        return wav
    # naive frame dropper for placeholder
    frame = int(0.02 * sr)
    y = wav.copy()
    for i in range(0, y.shape[-1], frame):
        if np.random.rand() < (loss_pct/100.0):
            y[..., i:i+frame] = 0.0
    return y
