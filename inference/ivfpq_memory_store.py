import torch
import faiss
import faiss.contrib.torch_utils

from typing import List

class IVFPQMemoryStore:
    """
    Works for key/value pairs type storage
    e.g., working and long-term memory
    """

    """
    An object group is created when new objects enter the video
    Objects in the same group share the same temporal extent
    i.e., objects initialized in the same frame are in the same group
    For DAVIS/interactive, there is only one object group
    For YouTubeVOS, there can be multiple object groups
    """

    def __init__(self, d_vector, n_centroid_ids, n_clusters, nprobe):

        # keys are stored in a single tensor and are shared between groups/objects
        # values are stored as a list indexed by object groups
        self.quantizer = faiss.IndexFlatL2(d_vector)
        cpu_index = faiss.IndexIVFPQ(self.quantizer, d_vector, n_clusters, n_centroid_ids, 8)
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        self.obj_groups = []
        # for debugging only
        self.all_objects = []

        # shrinkage and selection are also single tensors
        self.s = self.e = self.v = None

    def add(self, key, value, shrinkage, selection):

        # add data
        if not self.index.is_trained:
            self.index.train(key[0].transpose(0, 1).contiguous().float())
            self.index.add(key[0].transpose(0, 1).contiguous().float())
            self.s = shrinkage
            self.e = selection
            self.v = value
        else:
            self.index.add(key[0].transpose(0, 1).contiguous().float())
            self.v = torch.cat([self.v, value], -1)
            if shrinkage is not None:
                self.s = torch.cat([self.s, shrinkage], -1)
            if selection is not None:
                self.e = torch.cat([self.e, selection], -1)
    
    def topk(self, query, k=30):
        formatted_query = query[0].transpose(0, 1).contiguous().float()

        # returns a tuple of topk L2 similarity values and their indices
        return self.index.search(formatted_query, k).unsqueeze(0)

    def get_v_size(self, ni: int):
        return self.v[ni].shape[2]

    def engaged(self):
        return self.index.is_trained

    @property
    def size(self):
        return self.index.ntotal

    @property
    def num_groups(self):
        return len(self.v)

    @property
    def value(self):
        return self.v

    @property
    def shrinkage(self):
        return self.s

    @property
    def selection(self):
        return self.e

