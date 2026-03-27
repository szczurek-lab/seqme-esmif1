import numpy as np
import torch
from tqdm import tqdm


class ESMIF1:
    """
    Wrapper for the ESM inverse folding (ESM-IF1) model.

    Reference:
        Hsu et al., "Learning inverse folding from millions of predicted structures"
        (https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2)
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 256,
        verbose: bool = False,
    ):
        """
        Initialize the model.

        Args:
            device: Device to run inference on, e.g., ``"cuda"`` or ``"cpu"``.
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        import esm

        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def compute_perplexity(
        self,
        coordinates: list[np.ndarray],
        sequences: list[str],
    ) -> np.ndarray:
        """
        Compute perplexity after inverse folding the backbones (coordinates) and comparing against the target sequences.

        Args:
            coordinates: List of amino acid coordinates. Each entry: len(sequence) x 3 x 3 for N, CA, C atoms.
            sequences: Amino acid sequences associated with the coordinates (backbone).

        Returns:
            np.ndarray: Perplexity scores.
        """
        perplexities = []
        for i in tqdm(
            range(0, len(sequences), self.batch_size), disable=not self.verbose
        ):
            batch_coordinates = coordinates[i : i + self.batch_size]
            batch_sequences = sequences[i : i + self.batch_size]

            batch_perplexities = _compute_perplexity(
                model=self.model,
                alphabet=self.alphabet,
                coords=batch_coordinates,
                sequences=batch_sequences,
                device=self.device,
            )

            perplexities.append(batch_perplexities)

        return np.concatenate(perplexities)


# Adapted from: https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/util.py#L125
def _tokenize(alphabet, coords: list[np.ndarray], sequences: list[str], device: str):
    from esm.inverse_folding.util import CoordBatchConverter

    batch_converter = CoordBatchConverter(alphabet)
    coords, confidence, _, tokens, padding_mask = batch_converter.from_lists(
        coords_list=coords, seq_list=sequences, device=device
    )
    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]
    target_mask = target != alphabet.padding_idx

    return coords, padding_mask, confidence, prev_output_tokens, target, target_mask


def _logits_to_perplexity(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    token_nll = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1)
    return torch.exp(nll)


def _compute_perplexity(
    model,
    alphabet,
    coords: list[np.ndarray],
    sequences: list[str],
    device: str,
) -> np.ndarray:
    coords, padding_mask, confidence, prev_output_tokens, targets, target_mask = (
        _tokenize(alphabet=alphabet, coords=coords, sequences=sequences, device=device)
    )
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    perplexities = _logits_to_perplexity(logits, targets, target_mask).cpu()

    return perplexities.cpu().numpy()


def compute_perplexity(
    coordinates: list[np.ndarray],
    sequences: list[str],
    *,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    return ESMIF1(device=device, batch_size=batch_size).compute_perplexity(
        coordinates, sequences
    )


if __name__ == "__main__":
    coords = np.ones((4, 3, 3), dtype=np.float32)
    sequence = "AKMM"

    ppl = compute_perplexity([coords], [sequence])
    np.testing.assert_allclose(ppl, [30.44957], rtol=1e-5)
