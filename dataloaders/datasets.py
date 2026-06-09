# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications copyright (c) 2026 Matthias Nägele.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

import h5py
import numpy as np
from torch.utils import data


class BHACUniformDataset(data.Dataset):
    """
    Dataset for BHAC FNO-exported HDF5 files that contain:
      - fields: (nvars, nt, nx, ny)
      - varnames: (nvars,)
      - t: (nt,), x: (nx,), y: (ny,)
    """

    def __init__(
        self,
        data_path,
        output_names="output-????",
        file_name="fno_uniform_level1.h5",
        num_train=None,
        num_test=None,
        use_train=True,
    ):
        self.data_path = data_path
        self.output_names = output_names
        self.file_name = file_name
        self.use_train = use_train

        raw_path = os.path.join(data_path, output_names, file_name)
        files_raw = sorted(glob.glob(raw_path))
        self.files_raw = files_raw
        n = len(files_raw)
        if n == 0:
            raise FileNotFoundError(f"No files matched: {raw_path}")

        if (num_train is None) or (num_train > n):
            num_train = n
        self.num_train = num_train

        if (num_test is None) or (num_test > (n - num_train)):
            num_test = n - num_train
        self.num_test = num_test

        self.train_files = self.files_raw[:num_train]
        self.test_files = self.files_raw[num_train : num_train + num_test]
        self.files = self.train_files if use_train else self.test_files

    def __len__(self):
        return len(self.files)

    def _read_varnames(self, h5file):
        v = h5file["varnames"][:]
        # robust decode
        out = []
        for s in v:
            if isinstance(s, (bytes, np.bytes_)):
                out.append(s.decode("utf-8"))
            else:
                out.append(str(s))
        return out

    def __getitem__(self, index):
        # return filename; the dataloader will decide what slices/channels to read
        return self.files[index]

    def get_coords(self, index):
        file = self.files[index]
        with h5py.File(file, "r") as f:
            t = f["t"][:]
            x = f["x"][:]
            y = f["y"][:]
        return t, x, y

    def get_var_index_map(self, index=0):
        file = self.files[index]
        with h5py.File(file, "r") as f:
            names = self._read_varnames(f)
        return {name: i for i, name in enumerate(names)}
