"""Compact offline temporal models; no cross-period state or hosted inference."""
import numpy as np
import torch
from torch import nn
from solution.evidence import validate_blocks


class ResidualTemporal(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                                  nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.body(x) * .5


class TemporalNet(nn.Module):
    def __init__(self, kind, n_features, config):
        super().__init__(); self.kind = kind; self.config = dict(config)
        channels = int(config.get('channels', 32)); layers = int(config.get('layers', 5 if kind == 'tcn' else 1))
        dropout = float(config.get('dropout', .05))
        if channels < 1 or not 1 <= layers <= 8:
            raise ValueError('invalid temporal network size')
        if kind == 'tcn':
            self.input = nn.Conv1d(n_features, channels, 1)
            self.body = nn.Sequential(*(ResidualTemporal(channels, 2**i, dropout) for i in range(layers)))
            self.output = nn.Conv1d(channels, 3, 1)
            self.radius = 2 * sum(2**i for i in range(layers))
        elif kind == 'bilstm':
            self.input = nn.Linear(n_features, channels)
            self.body = nn.LSTM(channels, channels, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=dropout if layers > 1 else 0)
            self.output = nn.Linear(2 * channels, 3); self.radius = None
        elif kind == 'transformer':
            heads = int(config.get('heads', 4))
            if channels % heads:
                raise ValueError('attention width must be divisible by heads')
            self.input = nn.Linear(n_features, channels)
            self.position = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
            layer = nn.TransformerEncoderLayer(channels, heads, dim_feedforward=channels * 2,
                                               dropout=dropout, batch_first=True, activation='gelu')
            self.body = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
            self.output = nn.Linear(channels, 3); self.radius = None
        else:
            raise ValueError('unknown temporal model')

    def forward(self, x):
        if self.kind == 'tcn':
            return self.output(self.body(self.input(x.transpose(1, 2)))).transpose(1, 2)
        x = self.input(x)
        if self.kind == 'bilstm':
            x, _ = self.body(x)
        else:
            x = x + self.position(x.transpose(1, 2)).transpose(1, 2)
            x = self.body(x)
        return self.output(x)


def build_network(kind, n_features, config):
    torch.set_num_threads(4)
    return TemporalNet(kind, n_features, config)


def network_probabilities(model, x, blocks, chunk_size=2048):
    x = np.asarray(x, dtype=np.float32); validate_blocks(blocks, len(x))
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('invalid temporal features')
    model.cpu().eval(); result = np.full((len(x), 3), 1 / 3, dtype=np.float64)
    with torch.inference_mode():
        for start, stop in blocks:
            if model.kind == 'bilstm' or (chunk_size <= 0 and model.kind == 'tcn'):
                logits = model(torch.from_numpy(x[start:stop])[None])
                result[start:stop] = torch.softmax(logits, dim=-1)[0].numpy()
                continue
            width = int(model.config.get('window', 256)) if model.kind == 'transformer' else chunk_size
            context = int(model.config.get('context', 128)) if model.kind == 'transformer' else model.radius
            for a in range(start, stop, width):
                b = min(a + width, stop); low = max(start, a - context); high = min(stop, b + context)
                if model.kind == 'transformer':
                    # Fixed canonical windows are part of this model's definition.
                    batch = np.zeros((width + 2 * context, x.shape[1]), np.float32)
                    offset = low - (a - context)
                    batch[offset:offset + high - low] = x[low:high]
                    logits = model(torch.from_numpy(batch)[None])[0, context:context + b - a]
                else:
                    logits = model(torch.from_numpy(x[low:high])[None])[0, a - low:b - low]
                result[a:b] = torch.softmax(logits, dim=-1).numpy()
    return result


def train_network(kind, config, training, swap_index, seed):
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = build_network(kind, training[0]['x'].shape[1], config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('learning_rate', .001)),
                                  weight_decay=float(config.get('weight_decay', .001)))
    epochs = int(config.get('epochs', 8)); steps = int(config.get('steps_per_epoch', 64))
    batch_size = int(config.get('batch_size', 8)); window = int(config.get('window', 256))
    context = int(config.get('context', 128)); length = window + 2 * context
    if min(epochs, steps, batch_size, window) < 1 or context < 0:
        raise ValueError('invalid network training settings')
    weights = np.asarray([float(row.get('weight', 1)) for row in training]); weights /= weights.sum()
    losses = []
    for epoch in range(epochs):
        total = 0.; model.train()
        for step in range(steps):
            batch = np.zeros((batch_size, length, training[0]['x'].shape[1]), np.float32)
            target = np.full((batch_size, length), 3, dtype=np.int64)
            for i, chosen in enumerate(rng.choice(len(training), batch_size, p=weights)):
                row = training[chosen]
                blocks = row['blocks']; sizes = np.array([b - a for a, b in blocks], dtype=float)
                start, stop = blocks[rng.choice(len(blocks), p=sizes / sizes.sum())]
                anchor = int(rng.integers(start, stop))
                low = max(start, anchor - context); high = min(stop, anchor + window + context)
                offset = low - (anchor - context)
                batch[i, offset:offset + high - low] = row['x'][low:high]
                end = min(anchor + window, stop)
                target[i, context:context + end - anchor] = row['y'][anchor:end]
                if rng.random() < .5:
                    batch[i] = batch[i, :, swap_index].T
                    y = target[i]; y[y < 2] = 1 - y[y < 2]
            valid = target != 3
            if not valid.any():
                continue
            logits = model(torch.from_numpy(batch))
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), torch.from_numpy(target).reshape(-1), ignore_index=3)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step(); total += float(loss.detach())
        losses.append(total / steps)
        print(f'{kind} epoch {epoch + 1}/{epochs} loss {losses[-1]:.5f}', flush=True)
    model.eval()
    return model, losses
