import torch
import warnings

from inference.ivfpq_memory_store import IVFPQMemoryStore
from model.memory_util import *
from inference.kv_memory_store import KeyValueMemoryStore


class IVFPQManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """
    def __init__(self, config):
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The hidden state will be stored in a single tensor for all objects
        # B x num_objects x CH x H x W
        self.hidden = None

        self.mem = IVFPQMemoryStore(64, 8, 128, 32)
        self.kvmem = KeyValueMemoryStore(count_usage=True)

        self.reset_config = True

    def update_config(self, config):
        self.reset_config = True
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

    def _readout(self, affinity, v):
        # this function is for a single object group
        return v @ affinity

    def match_memory(self, query_key, selection):
        # query_key: B x C^k x H x W
        # selection:  B x C^k x H x W
        b, _, h, w = query_key.shape

        query_key = query_key.flatten(start_dim=2)
        selection = selection.flatten(start_dim=2) if selection is not None else None

        """
        Memory readout using keys
        """
        D, I = self.mem.topk(query_key, self.top_k)

        x_exp = D.nan_to_num().exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)

        torch.nn.functional.relu(I, inplace=True) # purely to remove -1s, needs solution

        affinity = torch.zeros(b, self.mem.v.shape[-1], h * w, device='cuda:0').scatter_(1, I, x_exp) # B*N*HW
        

        # Shared affinity within each group
        all_readout_mem = self._readout(affinity, self.mem.v)

        return all_readout_mem.view(all_readout_mem.shape[0], self.CV, h, w)

    def add_memory(self, key, shrinkage, value, objects, selection=None):
        # key: 1*C*H*W
        # value: 1*num_objects*C*H*W
        # objects contain a list of object indices
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H*self.W

        # key:   1*C*N
        # value: num_objects*C*N
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2) 
        value = value[0].flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        if selection is not None:
            selection = selection.flatten(start_dim=2)

        self.mem.add(key, value, shrinkage, selection, objects)


    def create_hidden_state(self, n, sample_key):
        # n is the TOTAL number of objects
        h, w = sample_key.shape[-2:]
        if self.hidden is None:
            self.hidden = torch.zeros((1, n, self.hidden_dim, h, w), device=sample_key.device)
        elif self.hidden.shape[1] != n:
            self.hidden = torch.cat([
                self.hidden, 
                torch.zeros((1, n-self.hidden.shape[1], self.hidden_dim, h, w), device=sample_key.device)
            ], 1)

        assert(self.hidden.shape[1] == n)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden