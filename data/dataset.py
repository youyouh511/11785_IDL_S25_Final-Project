import json
import torch
from torch.utils.data import Dataset, DataLoader

class JsonFireDataset(Dataset):
    def __init__(self, json_path, local_keys=None, oci_keys=None):
        """
        Args:
            json_path (str): path to your NDJSON file (one JSON object per line).
            local_keys (list of str): names of the local_variables channels, e.g. ['T2M','TP','VPD_CF'].
                                      If None, inferred from the first sample.
            oci_keys   (list of str): names of the ocis channels, e.g. ['NAO','NINA34_ANOM'].
                                      If None, inferred from the first sample.
        """
        # load all lines
        with open(json_path, 'r', encoding='utf-8') as f:
            self.records = [json.loads(line) for line in f]

        # infer channel order if not given
        first = self.records[0]
        if local_keys is None:
            local_keys = list(first['local_variables'].keys())
        if oci_keys is None:
            oci_keys = list(first['ocis'].keys())

        self.local_keys = local_keys
        self.oci_keys   = oci_keys
        self.channel_keys = self.local_keys + self.oci_keys

        # sanity-check that every record has the same length L
        L = len(first['local_variables'][self.local_keys[0]])
        for rec in self.records:
            assert all(len(rec['local_variables'][k]) == L for k in self.local_keys), "inconsistent L"
            assert all(len(rec['ocis'][k]) == L for k in self.oci_keys),       "inconsistent L"
        self.L = L

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # collect each channel sequence into a list
        seqs = []
        for k in self.local_keys:
            seqs.append(rec['local_variables'][k])
        for k in self.oci_keys:
            seqs.append(rec['ocis'][k])

        # stack → shape (channel, L)
        x = torch.tensor(seqs, dtype=torch.float32)
        y = torch.tensor(rec['target'], dtype=torch.long)
        return x, y