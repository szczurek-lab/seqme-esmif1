# seqme-ESM-IF1

Integration of the [ESM inverse-folding model](https://github.com/facebookresearch/esm) with seqme.

## Usage

Define the entry-point and call the model:

```python
import seqme as sm
import numpy as np

model = sm.models.ThirdPartyModel(
    entry_point="esmif1.model:compute_perplexity",
    path="../thirdparty/esmif1",
    url="https://github.com/szczurek-lab/seqme-esmif1",
    extras=["cpu"],  # Options: "cpu", "cu126", "cu128", "cu130"
)

coords = [np.ones((4, 3, 3), dtype=np.float32)]  # shape: (seq_len, atoms, xyz), atoms: N, CA, C
sequences = ["AKMM"]

model(coords, sequences, batch_size=256, device="cpu")
```

For CUDA, set `extras` to the appropriate version, e.g. `extras=["cu126"]`.

### End-to-end example

Fold sequences with ESMFold, then compute inverse-folding perplexity:

```python
sequences = ["MKRM", "KKRPR"]

folder = sm.models.ESMFold()  # Folding model
folds = folder.fold(sequences, convention="atom37", compute_ptm=False, return_type="dict")

atom_indices = [0, 1, 2]  # N, CA, C
coords = [seq_pos[:, atom_indices, :] for seq_pos in folds["positions"]]

# Inverse folding model
inv_folder = sm.models.ThirdPartyModel(
    entry_point="esmif1.model:compute_perplexity",
    path="../thirdparty/esmif1",
    url="https://github.com/szczurek-lab/seqme-esmif1",
    extras=["cpu"],  # Options: "cpu", "cu126", "cu128", "cu130"
)

inv_folder.compute_perplexity(coords, sequences)  # scPerplexity
```