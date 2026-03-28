# seqme-ESM-IF1

Integration of the [ESM inverse-folding model](https://github.com/facebookresearch/esm) with seqme.

## Functions

| Function | Description |
|---|---|
| `compute_perplexity(coords, sequences)` | Perplexity of native sequences given backbone coordinates |
| `compute_sequence_recovery(coords, sequences)` | Fraction of positions recovered by sampling, averaged over `n_samples` |
| `sample(coords)` | Sample sequences conditioned on backbone coordinates |

## Usage

Define the entry-point and call the model:

```python
import seqme as sm
import numpy as np

model = sm.models.ThirdPartyModel(
    entry_point="esmif1.model:compute_perplexity",
    path="../thirdparty/esmif1",
    url="https://github.com/szczurek-lab/seqme-esmif1",
)

coords = [np.ones((4, 3, 3), dtype=np.float32)]  # shape: (seq_len, atoms, xyz), atoms: N, CA, C
sequences = ["AKMM"]

model(coords, sequences, batch_size=256, device="cpu")
```


### End-to-end example

Fold sequences with ESMFold, then compute inverse-folding perplexity and sequence recovery:

```python
import seqme as sm

sequences = ["MKRM", "KKRPR"]

folder = sm.models.ESMFold()
folds = folder.fold(sequences, convention="atom37", compute_ptm=False, return_type="dict")

atom_indices = [0, 1, 2]  # N, CA, C
coords = [seq_pos[:, atom_indices, :] for seq_pos in folds["positions"]]

# Perplexity — lower is better; measures how well the structure explains the sequence
inv_folder = sm.models.ThirdPartyModel(
    entry_point="esmif1.model:compute_perplexity",
    path="../thirdparty/esmif1",
    url="https://github.com/szczurek-lab/seqme-esmif1",
)
perplexity = inv_folder(coords, sequences)

# Sequence recovery — fraction of residues reproduced by sampling; higher is better
seq_recovery = sm.models.ThirdPartyModel(
    entry_point="esmif1.model:compute_sequence_recovery",
    path="../thirdparty/esmif1",
    url="https://github.com/szczurek-lab/seqme-esmif1",
)
recovery = seq_recovery(coords, sequences, n_samples=10)

# Sample new sequences conditioned on backbone
sampler = sm.models.ThirdPartyModel(
    entry_point="esmif1.model:sample",
    path="../thirdparty/esmif1",
    url="https://github.com/szczurek-lab/seqme-esmif1",
)
sampled = sampler(coords)
```
