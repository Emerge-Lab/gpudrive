import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LSTM"]

class LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=False)

        for name, param in self.lstm.named_parameters():
            # LSTM parameters are named weight_* and bias_*
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        self.num_layers = num_layers
        self.hidden_shape = (2, self.num_layers, hidden_channels)

    def forward(self, in_features, cur_hidden):
        in_features = in_features.view(1, *in_features.shape)

        out, (new_h, new_c) = self.lstm(in_features,
                                        (cur_hidden[0], cur_hidden[1]))

        new_hidden = torch.stack([new_h, new_c], dim=0)

        return out.view(*out.shape[1:]), new_hidden

    def _get_lstm_params(self, layer_idx):
        weight_ih = getattr(self.lstm, f'weight_ih_l{layer_idx}')
        bias_ih = getattr(self.lstm, f'bias_ih_l{layer_idx}')

        weight_hh = getattr(self.lstm, f'weight_hh_l{layer_idx}')
        bias_hh = getattr(self.lstm, f'bias_hh_l{layer_idx}')

        return weight_ih, bias_ih, weight_hh, bias_hh

    def lstm_iter_slow(self, layer_idx, in_features, cur_hidden, breaks):
        weight_ih, bias_ih, weight_hh, bias_hh = self._get_lstm_params(
                layer_idx)

        ifgo = (
            F.linear(in_features, weight_ih, bias_ih) +
            F.linear(cur_hidden[0, :, :], weight_hh, bias_hh)
        )

        hs = self.hidden_shape[-1] # hidden feature size

        c = (F.sigmoid(ifgo[:, hs:2*hs]) * cur_hidden[1, :, :] +
            F.sigmoid(ifgo[:, 0:hs]) * F.tanh(ifgo[:, 2*hs:3*hs]))

        o = ifgo[:, 3*hs:4*hs]

        h = o * F.tanh(c)

        new_hidden = torch.stack([h, c], dim=0)

        return o, new_hidden

    # Manually written LSTM implementation, doesn't work
    def fwd_sequence_slow(self, in_sequences, start_hidden, sequence_breaks):
        seq_len = in_sequences.shape[0]

        hidden_dim_per_layer = start_hidden.shape[-1]

        zero_hidden = torch.zeros((2, self.num_layers, 1,
                                   hidden_dim_per_layer),
                                  device=start_hidden.device,
                                  dtype=start_hidden.dtype)

        out_sequences = []

        cur_hidden = start_hidden
        for i in range(seq_len):
            cur_features = in_sequences[i]
            cur_breaks = sequence_breaks[i]

            new_hiddens = []
            for layer_idx in range(self.num_layers):
                cur_features, new_hidden = self.lstm_iter_slow(
                    layer_idx, cur_features, cur_hidden[:, layer_idx, :, :],
                    sequence_breaks[i])

                new_hiddens.append(new_hidden)
                out_sequences.append(cur_features)

            cur_hidden = torch.stack(new_hiddens, dim=1)

            cur_hidden = torch.where(
                cur_breaks.bool(), zero_hidden, cur_hidden)

        return torch.stack(out_sequences, dim=0)

    # Just call forward repeatedly
    def fwd_sequence_default(self, in_sequences, start_hidden, sequence_breaks):
        seq_len = in_sequences.shape[0]

        hidden_dim_per_layer = start_hidden.shape[-1]

        zero_hidden = torch.zeros((2, self.num_layers, 1,
                                   hidden_dim_per_layer),
                                  device=start_hidden.device,
                                  dtype=start_hidden.dtype)

        out_sequences = []

        cur_hidden = start_hidden
        for i in range(seq_len):
            cur_features = in_sequences[i]
            cur_breaks = sequence_breaks[i]

            out, new_hidden = self.forward(cur_features, cur_hidden)
            out_sequences.append(out)

            cur_hidden = torch.where(
                cur_breaks.bool(), zero_hidden, new_hidden)

        return torch.stack(out_sequences, dim=0)

    fwd_sequence = fwd_sequence_default
