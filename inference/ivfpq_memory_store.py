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
        self.s = self.e
        self.v = []

    def add(self, key, value, shrinkage, selection, objects):

        # add keys
        if not self.index.is_trained:
            self.index.train(key[0].transpose(0, 1).contiguous().float())
            self.index.add(key[0].transpose(0, 1).contiguous().float())
            self.s = shrinkage
            self.e = selection
        else:
            self.index.add(key[0].transpose(0, 1).contiguous().float())
            if shrinkage is not None:
                self.s = torch.cat([self.s, shrinkage], -1)
            if selection is not None:
                self.e = torch.cat([self.e, selection], -1)

        # add values
        if objects is not None:
            # When objects is given, v is a tensor; used in working memory
            assert isinstance(value, torch.Tensor)
            # First consume objects that are already in the memory bank
            # cannot use set here because we need to preserve order
            # shift by one as background is not part of value
            remaining_objects = [obj-1 for obj in objects]
            for gi, group in enumerate(self.obj_groups):
                for obj in group:
                    # should properly raise an error if there are overlaps in obj_groups
                    remaining_objects.remove(obj)
                self.v[gi] = torch.cat([self.v[gi], value[group]], -1)

            # If there are remaining objects, add them as a new group
            if len(remaining_objects) > 0:
                new_group = list(remaining_objects)
                self.v.append(value[new_group])
                self.obj_groups.append(new_group)
                self.all_objects.extend(new_group)
        else:
            # When objects is not given, v is a list that already has the object groups sorted
            # used in long-term memory
            assert isinstance(value, list)
            for gi, gv in enumerate(value):
                if gv is None:
                    continue
                if gi < self.num_groups:
                    self.v[gi] = torch.cat([self.v[gi], gv], -1)
                else:
                    self.v.append(gv)

        
    
    def topk(self, query, k=30):
        formatted_query = query[0].transpose(0, 1).contiguous().float()

        # returns a tuple of topk L2 similarity values and their indices
        D, I = self.index.search(formatted_query, k)
        return D.transpose(0, 1).unsqueeze(0), I.transpose(0, 1).unsqueeze(0)

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

