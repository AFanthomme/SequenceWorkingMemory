import torch as tch 
from torch.nn import Module, Linear, ReLU, Identity, Parameter

device = tch.device('cuda:0')

class Decoder(Module):
    def __init__(self, in_size=1024, state_size=64):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.state_size = state_size

        self.in_layer = Linear(in_size, state_size)
        self.rec_layer = Linear(state_size, state_size)
        # self.rec_layer2 = Linear(state_size, state_size)
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
    def __init__(self, in_size=64, role_size=32, filler_size=None, T=3, device=device):
        super(TensorSequenceEncoder, self).__init__()
        self.in_size = in_size
        self.role_size = role_size

        if filler_size is None:
            self.filler_size = self.role_size
        else:
            self.filler_size = filler_size

        self.device = device
        self.T = T
        # tmp = tch.randn(T, role_size).to(self.device) / tch.sqrt(role_size)
        # self.role_bindings = Parameter(tmp, requires_grad=False)

        self.role_rep_net = Linear(T, self.role_size).to(self.device)
        self.role_rep_net.weight.requires_grad = False
        self.role_rep_net.bias.requires_grad = False
        
        self.filler_rep_net = Linear(self.in_size, self.filler_size).to(self.device)
        self.filler_rep_net.weight.requires_grad = False
        self.filler_rep_net.bias.requires_grad = False
        self.out_size = self.role_size * self.filler_size
        self.state_size = self.out_size

    def forward(self, x):
        x = x.to(self.device)
        T = x.shape[1]
        bs = x.shape[0]
        out = tch.zeros(bs, T, self.out_size).to(self.device)

        ranks = tch.zeros(bs, T, T).to(self.device)
        for t in range(T):
            ranks[:, t, t] = 1 



        # print( self.filler_rep_net(x[:,0,:]))

        # tmp = tch.zeros(bs, self.out_size).to(self.device)
        # for t in range(T):
        #     # tmp = tmp + tch.einsum('p, bq->bpq', self.role_bindings[t], self.filler_rep_net(x[:,t,:])).view(bs, -1)
        #     tmp = tmp + tch.einsum('p, bq->bpq', self.role_rep_net(ranks), self.filler_rep_net(x[:,t,:])).view(bs, -1)
        # for t in range(T):
        #     out[:, t, :] = tmp

        # return out
            
        role_reps = self.role_rep_net(ranks)
        filler_reps = self.filler_rep_net(x)
        outer_products = tch.einsum('btr, btf->btrf', role_reps, filler_reps).view(bs, T, -1)
        tensor_product_rep = tch.sum(outer_products, dim=1).unsqueeze(dim=1).repeat(1, T, 1) # (bs, T, state_size)
        return tensor_product_rep

class RNNSequenceEncoder(Module):
    def __init__(self, in_size=128, state_size=1024, out_size=None, **kwargs):
        super(RNNSequenceEncoder, self).__init__()
        self.in_size = in_size
        self.state_size = state_size
        
        if out_size is not None:
            self.out_size = out_size

        self.in_layer = Linear(in_size, state_size)
        self.rec_layer = Linear(state_size, state_size)
        self.out_layer = Linear(state_size, state_size)
        self.activation = ReLU()

        self.device = device
        self.to(self.device)


    def get_underlying_rep(self, x):
        x = x.to(self.device)
        T = x.shape[1]
        bs = x.shape[0]
        internal_rep = tch.zeros(bs, T, self.state_size).to(self.device)

        state = tch.zeros(bs, self.state_size).to(self.device)
        for t in range(T):
            ext = self.in_layer(x[:,t,:])
            state = self.activation(self.rec_layer(state + ext))
            # state = self.activation(self.rec_layer2(state))

        for t in range(T):
            internal_rep[:, t, :] = state

        return out

    def forward(self, x):
        x = x.to(self.device)
        T = x.shape[1]
        bs = x.shape[0]
        out = tch.zeros(bs, T, self.state_size).to(self.device)

        state = tch.zeros(bs, self.state_size).to(self.device)
        for t in range(T):
            ext = self.in_layer(x[:,t,:])
            state = self.activation(self.rec_layer(state + ext))
            # state = self.activation(self.rec_layer2(state))

        plop = self.out_layer(state)
        for t in range(T):
            out[:, t, :] = plop

        return out


class TensorProductDecompositionNetwork(Module):
    "Map a sequence of encodings into a Tensor encoding of said sequence"
    def __init__(self, in_size=64, role_size=64, filler_size=None, out_size=512, T=3, n_dots=16, device=device, force_identity_out=False):
        super(TensorProductDecompositionNetwork, self).__init__()
        self.n_dots = n_dots
        self.in_size = in_size
        self.role_size = role_size

        if filler_size is None:
            self.filler_size = self.role_size
        else:
            self.filler_size = filler_size

        self.device = device
        self.role_representer = Linear(T, self.role_size)
        self.filler_representer = Linear(in_size, self.filler_size)
        self.out_size = out_size
        self.state_size = self.filler_size * self.role_size
        self.force_identity_out = force_identity_out

        if not self.force_identity_out:
            self.out_layer = Linear(self.state_size, self.out_size)
        else:
            assert self.out_size == self.state_size, "output size should be equal to product of role/filler sizes"
            self.out_layer = Identity()

        self.to(self.device)

    def get_underlying_TP(self, x):
        x = x.to(self.device) # (bs, T, in_size)
        T = x.shape[1]
        bs = x.shape[0] 
        out = tch.zeros(bs, T, self.out_size).to(self.device)

        # One-hot encoded rank information
        ranks = tch.zeros(bs, T, T).to(self.device)
        ranks[:, 0, 0] = 1 
        ranks[:, 1, 1] = 1 
        ranks[:, 2, 2] = 1 

        # Compute representations
        role_reps = self.role_representer(ranks) # (bs, T, role_size)
        filler_reps = self.filler_representer(x) # (bs, T, filler_size)

        # Binding role/fillers, summing over all roles
        outer_products = tch.einsum('btr, btf->btrf', role_reps, filler_reps).view(bs, T, -1)
        tensor_product_rep = tch.sum(outer_products, dim=1).unsqueeze(dim=1).repeat(1, T, 1) # (bs, T, state_size)

        return tensor_product_rep

    def forward(self, x):
        x = x.to(self.device) # (bs, T, in_size)
        T = x.shape[1]
        bs = x.shape[0] 
        out = tch.zeros(bs, T, self.out_size).to(self.device)

        # One-hot encoded rank information
        ranks = tch.zeros(bs, T, T).to(self.device)
        ranks[:, 0, 0] = 1 
        ranks[:, 1, 1] = 1 
        ranks[:, 2, 2] = 1 

        # Compute representations
        role_reps = self.role_representer(ranks) # (bs, T, role_size)
        filler_reps = self.filler_representer(x) # (bs, T, filler_size)

        # Binding role/fillers, summing over all roles
        outer_products = tch.einsum('btr, btf->btrf', role_reps, filler_reps).view(bs, T, -1)
        tensor_product_rep = tch.sum(outer_products, dim=1).unsqueeze(dim=1).repeat(1, T, 1) # (bs, T, state_size)

        return self.out_layer(tensor_product_rep)

if __name__ == '__main__':
    from environment import CircularDots
    env = CircularDots(n_dots=6, T=3)
    encodings, positions, indices = env.get_sequences(bs=10)
    print(positions.shape)

    decoder = Decoder()
    out = decoder(encodings)
    print(out.shape)