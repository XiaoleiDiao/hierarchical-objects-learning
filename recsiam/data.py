"""
module containing utilities to load
the dataset for the training
of the siamese recurrent network.
"""
import json
import copy
import functools
import logging
import lz4.frame

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Subset
from skimage import io
import torch

import recsiam.utils as utils

# entry
# { "id" :  int,
#   "paths" : str,
#   "metadata" : str}


def nextid():
    cid = 0
    while True:
        yield cid
        cid += 1


def descriptor_from_filesystem(root_path):
    desc = []
    root_path = Path(root_path)

    id_gen = nextid()

    embedded = False
    sample = next(root_path.glob("*/*/*")).name
    if sample.endswith("npy") or sample.endswith("lz4"):
        embedded = True

    for subd in sorted(root_path.iterdir()):

        if subd.is_dir():
            obj_desc = {"id":  next(id_gen), "name": str(subd.name)}

            if not embedded:
                seqs_dir = sorted(str(subsub)
                                  for subsub in subd.iterdir()
                                  if subsub.is_dir())
                seqs = [sorted(str(frame)
                               for frame in Path(d).iterdir())
                        for d in seqs_dir]
            else:
                seqs = sorted(str(subsub / sample)
                              for subsub in subd.iterdir()
                              if subsub.is_dir())

            obj_desc["paths"] = seqs

            desc.append(obj_desc)

    info_path = root_path / "metadata.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
    else:
        info = {}

    return info, desc


class VideoDataSet(Dataset):
    """
    Class that implements the pythorch Dataset
    of sequences of frames
    """
    def __init__(self, descriptor):

        self.logger = logging.getLogger("recsiam.data.VideoDataSet")

        self.descriptor = descriptor

        if not isinstance(self.descriptor, (list, tuple, np.ndarray)):
            with Path(self.descriptor).open("r") as ifile:
                self.descriptor = json.load(ifile)
        self.info = self.descriptor[0]
        self.data = np.asarray(self.descriptor[1])

        self.paths = np.array([d["paths"] for d in self.data])
        self.seq_number = np.array([len(path) for path in self.paths])

        def get_id_entry(elem_id):
            return self.data[elem_id]["id"]

        self.id_table = np.vectorize(get_id_entry)

        self.embedded = False
        self.compressed = False
        try:
            np.load(self.paths[0][0])
            self.embedded = True
        except Exception:
            pass

        try:
            with lz4.frame.open(self.paths[0][0], mode="rb") as f:
                np.load(f)
                self.embedded = True
                self.compressed = True
        except Exception:
            pass

        self.n_elems = len(self.paths)

    @property
    def is_embed(self):
        return self.embedded

    def get_metadata(self, key, elem_ind, object_level=True):
        elem_ind = np.asarray(elem_ind)

        obj_d = self.data[elem_ind[0]]

        if object_level:
            val = obj_d[key]
        else:
            val = obj_d[key][elem_ind[1]]
        log = logging.getLogger(self.logger.name + ".get_metadata")
        log_str = "key = {}\telem_ind ={}\tval = {}".format(key, elem_ind, val)
        log.debug(log_str)

        return val

    def load_array(self, path):
        if not self.compressed:
            loaded = np.load(str(path))

        else:
            with lz4.frame.open(str(path), mode="rb") as f:
                loaded = np.load(f)

        return loaded

    def __len__(self):
        return self.n_elems

    def __getitem__(self, value):
        return self._getitem(value)

    def _getitem(self, value):

        if isinstance(value, (list, tuple, np.ndarray)):
            if self._valid_t(value):
                return self._get_single_item(*value)
            elif np.all([self._valid_t(val) for val in value]):
                return np.array([self._get_single_item(*val) for val in value])
            else:
                raise TypeError("Invalid argument type: {}.".format(value))
        else:
            raise TypeError("Invalid argument type: {}.".format(value))

    @staticmethod
    def _valid_t(value):
        return isinstance(value, (tuple, list, np.ndarray)) and \
                len(value) == 3 and \
                isinstance(value[0], (int, np.integer)) and \
                isinstance(value[1], (int, np.integer)) and \
                isinstance(value[2], (int, np.integer,
                                      slice, list, np.ndarray))

    def sample_size(self):
        return self._get_single_item(0, 0, slice(0, 1)).shape[1:]

    def _get_single_item(self, idx1, idx2, idx3):

        path = self.paths[idx1]
        seq_path = path[idx2]
        if not self.is_embed:
            p_list = np.array(seq_path)[idx3]
        else:
            p_list = seq_path, idx3

        l = logging.getLogger(self.logger.name + "._get_single_item")
        l.debug("path_ind = {}\tseq_path ={}\telem_ind = {}".format(idx1, seq_path, idx3))
        sequences = self._load_sequence(p_list)

        return sequences

    def _load_sequence(self, paths_list):

        if not self.is_embed:
            sequence = np.array(io.imread_collection(paths_list,
                                                     conserve_memory=False))
            # if gray, add bogus dimension
            if len(sequence.shape) == 2 + int(len(paths_list) > 1):
                sequence = sequence[..., None]
            sequence = np.transpose(sequence, (0, 3, 1, 2))

        else:
            sequence = self.load_array(paths_list[0])[paths_list[1]]

        return sequence

    def gen_embed_dataset(self):
        for obj in range(self.n_elems):
            for seq in range(len(self.paths[obj])):
                yield self[obj, seq, :], self.paths[obj][seq]


def dataset_from_filesystem(root_path):
    descriptor = descriptor_from_filesystem(root_path)
    return VideoDataSet(descriptor)


class TrainSeqDataSet(VideoDataSet):

    def __getitem__(self, value):
        if isinstance(value, (list, tuple, np.ndarray)) and \
           len(value) == 2 and \
           np.all([self._valid_t(val) for val in value]):

            items = self._getitem(value)
            seq_len = np.array([len(val)
                                for val in items])

            return items, seq_len, (value[0][0], value[1][0])
        else:
            error_str = "The input must be in the form " +\
                        "((int, int, slice), (int, int,  slice)). " +\
                        "Found {}"

            raise ValueError(error_str.format(value))


class FlattenedDataSet(VideoDataSet):

    def __init__(self, *args, preload=False, by_frame=False):
        super().__init__(*args)

        self.by_frame = by_frame
        self.val_map = []

        if not self.by_frame:
            for itx in range(len(self.seq_number)):
                self.val_map.extend([(itx, i) for i in range(self.seq_number[itx])])
        else:
            for itx in range(len(self.seq_number)):
                for i in range(self.seq_number[itx]):
                    self.val_map.extend([(itx, i, j)
                                         for j in range(super().__getitem__((itx, i, slice(None))).shape[0])])

        self.val_map = np.array(self.val_map)

        self.flen = len(self.val_map)

        self.preloaded = None
        if preload:
            self.preloaded = []
            for i in range(len(self)):
                self.preloaded.append(self.real_getitem(i))

            self.preloaded = np.array(self.preloaded)

    def map_value(self, value):
        return self.val_map[value]

    def __len__(self):
        return self.flen

    def get_metadata(self, key, elem_ind, **kwargs):
        if not self.by_frame:
            return super().get_metadata(key, self.val_map[elem_ind].squeeze(), **kwargs)
        else:
            return super().get_metadata(key, self.val_map[elem_ind,:2].squeeze(), **kwargs)

    def get_label(self, value):
        if type(value) == slice:
            return self.map_value(value)[:, 0]

        ndim = np.ndim(value)
        if ndim == 0:
            return self.map_value(value)[0]
        elif ndim == 1:
            return self.map_value(value)[:, 0]
        else:
            raise ValueError("np.ndim(value) > 1")

    def __getitem__(self, i):
        if self.preloaded is not None:
            return self.preloaded[i]
        else:
            return self.getitems(i)

    def getitems(self, ind):
        if isinstance(ind, slice):
            ind = np.arange(*ind.indices(len(self)))

        if isinstance(ind, (list, tuple, np.ndarray)):
            return np.array([self.real_getitem(i) for i in ind])
        else:
            return self.real_getitem(ind)

    def real_getitem(self, value):
        if not self.by_frame:
            t = tuple(self.map_value(value)) + (slice(None),)
        else:
            t = self.map_value(value)

        items = super().__getitem__(t)
        return items, t[0]

    def balanced_sample(self, elem_per_class, rnd, separate=False, ind_subset=None):
        if self.by_frame:
            raise NotImplementedError()
        if ind_subset is None:
            p_ind = rnd.permutation(len(self.val_map))
        else:
            assert np.unique(ind_subset).size == ind_subset.size
            assert (ind_subset >= 0).all() and (ind_subset < len(self.val_map)).all()
            p_ind = rnd.permutation(ind_subset)
        perm = self.val_map[p_ind]
        cls = perm[:, 0]

        _, indices = np.unique(cls, return_index=True)

        remaining_ind = np.delete(np.arange(len(cls)), indices)

        ind_sets = [indices]

        for i in range(elem_per_class - 1):
            p = cls[remaining_ind]
            _, ind = np.unique(p, return_index=True)

            ind_sets.append(ind)
            indices = np.concatenate([indices, remaining_ind[ind]])
            remaining_ind = np.delete(remaining_ind, ind)

        if not separate:
            return p_ind[indices]
        else:
            return tuple(p_ind[i] for i in ind_sets)

    def get_n_objects(self, number, rnd, ind_subset=None):
        if self.by_frame:
            raise NotImplementedError()
        if ind_subset is None:
            elems = len(self.seq_number)
        else:
            elems = np.unique(self.get_label(ind_subset))
        obj_ind = rnd.choice(elems, size=number, replace=False)

        class_ind = np.where(np.isin(self.val_map[:, 0], obj_ind))[0]
        if ind_subset is not None:
            class_ind = np.intersect1d(class_ind, ind_subset)

        return class_ind


def list_collate(data):
    emb = [utils.astensor(d[0]) for d in data]
    lab = np.array([d[1] for d in data])

    return emb, lab


class ExtendedSubset(Subset):

    def __init__(self, dataset, indices=None):
        if indices is None:
            indices = np.arange(len(dataset))
        if isinstance(dataset, Subset):
            indices = dataset.indices[indices]
            dataset = dataset.dataset

        self.info = dataset.info
        super().__init__(dataset, indices)

    def get_label(self, value):
        return self.dataset.get_label(self.indices[value])

    def get_metadata(self, key, elem_ind, **kwargs):
        return self.dataset.get_metadata(key, self.indices[elem_ind], **kwargs)

    def split_balanced(self, elem_per_class, rnd):
        ind = self.dataset.balanced_sample(elem_per_class, rnd, False, self.indices)

        other_ind = np.setdiff1d(self.indices, ind)

        return (ExtendedSubset(self.dataset, ind),
                ExtendedSubset(self.dataset, other_ind))

    def split_n_objects(self, number, rnd):
        ind = self.dataset.get_n_objects(number, rnd, self.indices)
        other_ind = np.setdiff1d(self.indices, ind)

        return (ExtendedSubset(self.dataset, ind),
                ExtendedSubset(self.dataset, other_ind))

    def same_subset(self, dataset):
        assert len(dataset) == len(self.dataset)
        return ExtendedSubset(dataset, self.indices)



def train_shuf(dataset, seed, dl_args={}, setting=None):
    rs = np.random.RandomState
    rnd_s, rnd_e, rnd_i = [rs(s) for s in rs(seed).randint(2**32 - 1, size=3)]
    if setting is None:
        train_ind = rnd_s.permutation(np.arange(len(dataset)))
    elif setting["type"] == "tree":
        assert "hierarchy" in dataset.info
        names = [dataset.get_metadata("name", i)
                 for i in np.arange(len(dataset))]
        train_ind = utils.shuffle_tree_by_distance(rnd_s,
                                                   dataset.info["hierarchy"],
                                                   names,
                                                   **setting["setting_args"])

    train_ds = ExtendedSubset(dataset, train_ind)
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=False, collate_fn=list_collate, **dl_args)

    return train_dl, None, None


def train_factory(train_desc, test_seed, dl_args={}, ds_args={},
                  setting=None):

    train_ds = FlattenedDataSet(train_desc, **ds_args)

    return functools.partial(train_shuf, train_ds,
                             dl_args=dl_args, setting=setting)
