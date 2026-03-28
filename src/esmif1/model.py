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
        seed: int = 0,
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

        self.seed = seed

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
            range(0, len(sequences), self.batch_size),
            disable=not self.verbose,
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

    @torch.inference_mode()
    def compute_sequence_recovery(
        self,
        coordinates: list[np.ndarray],
        sequences: list[str],
        n_samples: int = 1,
    ) -> np.ndarray:
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        recovery = []
        for i in tqdm(
            range(0, len(sequences), self.batch_size),
            disable=not self.verbose,
        ):
            batch_coordinates = coordinates[i : i + self.batch_size]
            batch_sequences = sequences[i : i + self.batch_size]

            batch_recovery = np.zeros(len(batch_sequences))
            for _ in range(n_samples):
                generated = self.sample(batch_coordinates, generator=generator)
                for j, (native_seq, sampled_seq) in enumerate(
                    zip(batch_sequences, generated)
                ):
                    batch_recovery[j] += np.mean(
                        np.frombuffer(native_seq.encode(), dtype=np.uint8)
                        == np.frombuffer(sampled_seq.encode(), dtype=np.uint8)
                    )
            batch_recovery /= n_samples

            recovery.append(batch_recovery)
        return np.concatenate(recovery)

    @torch.inference_mode()
    def sample(
        self,
        coordinates: list[np.ndarray],
        *,
        generator: torch.Generator | None = None,
    ) -> list[str]:
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)

        samples = []
        for i in tqdm(
            range(0, len(coordinates), self.batch_size),
            disable=not self.verbose,
        ):
            batch_coordinates = coordinates[i : i + self.batch_size]

            batch_sequences = _sample(
                model=self.model,
                alphabet=self.alphabet,
                coords=batch_coordinates,
                device=self.device,
                generator=generator,
            )

            samples.extend(batch_sequences)
        return samples


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


def _sample(
    model,
    alphabet,
    coords: list[np.ndarray],
    generator: torch.Generator,
    partial_seq=None,
    temperature: float = 1.0,
    confidence: np.ndarray = None,
    device: str = None,
) -> list[str]:
    return [
        _sample_single(
            model,
            alphabet,
            seq_coords,
            generator,
            partial_seq,
            temperature,
            confidence,
        )
        for seq_coords in coords
    ]


def _sample_single(
    model,
    alphabet,
    coords,
    generator,
    partial_seq=None,
    temperature=1.0,
    confidence=None,
) -> list[str]:
    """
    Samples sequences based on multinomial sampling (no beam search).

    Args:
        coords: L x 3 x 3 list representing one backbone
        partial_seq: Optional, partial sequence with mask tokens if part of
            the sequence is known
        temperature: sampling temperature, use low temperature for higher
            sequence recovery and high temperature for higher diversity
        confidence: optional length L list of confidence scores for coordinates
    """
    from esm.inverse_folding.util import CoordBatchConverter
    import torch.nn.functional as F

    L = len(coords)
    # Convert to batch format
    batch_converter = CoordBatchConverter(alphabet)
    batch_coords, confidence, _, _, padding_mask = batch_converter(
        [(coords, confidence, None)]
    )

    # Start with prepend token
    mask_token_id = alphabet.get_idx("<mask>")
    sampled_tokens = torch.full((1, 1 + L), mask_token_id, dtype=int)
    sampled_tokens[0, 0] = alphabet.get_idx("<cath>")
    if partial_seq is not None:
        for i, c in enumerate(partial_seq):
            sampled_tokens[0, i + 1] = alphabet.get_idx(c)

    # Save incremental states for faster sampling
    incremental_state = dict()

    # Run encoder only once
    encoder_out = model.encoder(batch_coords, padding_mask, confidence)

    # Decode one token at a time
    for i in range(1, L + 1):
        if sampled_tokens[0, i] != mask_token_id:
            continue
        logits, _ = model.decoder(
            sampled_tokens[:, :i],
            encoder_out,
            incremental_state=incremental_state,
        )
        logits = logits[0].transpose(0, 1)
        logits /= temperature
        probs = F.softmax(logits, dim=-1)
        sampled_tokens[:, i] = torch.multinomial(probs, 1, generator=generator).squeeze(
            -1
        )
    sampled_seq = sampled_tokens[0, 1:]

    # Convert back to string via lookup
    return "".join([alphabet.get_tok(a) for a in sampled_seq])


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


def compute_sequence_recovery(
    coordinates: list[np.ndarray],
    sequences: list[str],
    *,
    n_samples: int = 1,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    return ESMIF1(device=device, batch_size=batch_size).compute_sequence_recovery(
        coordinates, sequences, n_samples=n_samples
    )


def sample(
    coordinates: list[np.ndarray],
    *,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    return ESMIF1(device=device, batch_size=batch_size).sample(coordinates)


if __name__ == "__main__":
    coords = np.ones((4, 3, 3), dtype=np.float32)
    sequence = "AKMM"

    n = 3
    all_coords = [coords] * n
    all_sequences = [sequence] * n

    ppl = compute_perplexity(all_coords, all_sequences)
    np.testing.assert_allclose(ppl, [30.44957] * n, rtol=1e-5)

    sequences = sample(all_coords)
    assert sequences == ["MGLR", "MIAG", "MPLN"]

    recovery = compute_sequence_recovery(all_coords, all_sequences)
    np.testing.assert_allclose(recovery, [0.0] * n, rtol=1e-5)
