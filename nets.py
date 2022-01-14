import torch as tch 
from torch.nn import Module, Linear, ReLU

device = tch.device('cuda:0')

class Decoder(Module):
    def __init__(self, in_size=4096, state_size=256):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.state_size = state_size

        self.in_layer = Linear(in_size, state_size)
        self.rec_layer = Linear(state_size, state_size)
        self.rec_layer2 = Linear(state_size, state_size)
        self.out_layer = Linear(state_size, state_size//2)
        self.out_layer2 = Linear(state_size//2, 2)
        self.activation = ReLU()

        self.device = device
        self.to(self.device)


    def forward(self, x):
        x = x.to(self.device)
        T = x.shape[1]
        bs = x.shape[0]
        out = tch.zeros(bs, T, 2).to(self.device)

        state = tch.zeros(bs, self.state_size).to(self.device)
        for t in range(T):
            ext = self.in_layer(x[:,t,:])
            state = self.activation(self.rec_layer(state + ext))
            # state = self.activation(self.rec_layer(state + ext))
            out[:, t, :] = self.out_layer2(self.activation(self.out_layer(state)))

        return out

class TensorSequenceEncoder(Module):
    "Map a sequence of encodings into a Tensor encoding of said sequence"
    def __init__(self, in_size=64, role_size=64, state_size=128, T=3, device=device):
        super(TensorSequenceEncoder, self).__init__()
        self.in_size = in_size
        self.state_size = state_size
        self.role_size = role_size
        self.device = device
        self.role_bindings = tch.randn(T, role_size).to(self.device)
        self.out_size = self.in_size * self.role_size

    def forward(self, x):
        x = x.to(self.device)
        T = x.shape[1]
        bs = x.shape[0]
        out = tch.zeros(bs, T, self.out_size).to(self.device)

        tmp = tch.zeros(bs, self.out_size).to(self.device)
        for t in range(T):
            tmp = tmp + tch.einsum('p, bq->bpq', self.role_bindings[t], x[:,t,:]).view(bs, -1)
            
        for t in range(T):
            out[:, t, :] = tmp

        return out

class RNNSequenceEncoder(Module):
    def __init__(self, in_size=128, state_size=1024, **kwargs):
        super(RNNSequenceEncoder, self).__init__()
        self.in_size = in_size
        self.state_size = state_size

        self.in_layer = Linear(in_size, state_size)
        self.rec_layer = Linear(state_size, state_size)
        self.rec_layer2 = Linear(state_size, state_size)
        self.activation = ReLU()

        self.device = device
        self.to(self.device)


    def forward(self, x):
        x = x.to(self.device)
        T = x.shape[1]
        bs = x.shape[0]
        out = tch.zeros(bs, T, self.state_size).to(self.device)

        state = tch.zeros(bs, self.state_size).to(self.device)
        for t in range(T):
            ext = self.in_layer(x[:,t,:])
            state = self.activation(self.rec_layer(state + ext))
            state = self.activation(self.rec_layer2(state))

        for t in range(T):
            out[:, t, :] = state

        return out


if __name__ == '__main__':
    from environment import CircularDots
    env = CircularDots(n_dots=6, T=3)
    encodings, positions, indices = env.get_sequences(bs=10)
    print(positions.shape)

    decoder = Decoder()
    out = decoder(encodings)
    print(out.shape)