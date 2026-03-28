import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


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
            device: Device to run inference on, e.g., ``"cuda"`` or ``"cpu"``. Defaults to CUDA if available.
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
            seed: Random seed for reproducible sampling.
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

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def compute_perplexity(
        self,
        coordinates: list[np.ndarray],
        sequences: list[str],
    ) -> np.ndarray:
        """
        Compute per-sequence perplexity of native sequences given backbone coordinates.

        Lower perplexity indicates the structure better explains the sequence. Sequences
        are scored by running the encoder over the backbone and computing cross-entropy
        of the decoder against the native tokens.

        Args:
            coordinates: Backbone coordinates, one array per sequence. Each array has
                shape ``(L, 3, 3)`` for N, CA, C atoms.
            sequences: Native amino acid sequences corresponding to each backbone.

        Returns:
            Perplexity score for each sequence, shape ``(N,)``.
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
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Estimate sequence recovery by sampling from the model and comparing against native sequences.

        For each backbone, ``n_samples`` sequences are drawn and the fraction of positions
        matching the native sequence is averaged across samples.

        Args:
            coordinates: Backbone coordinates, one array per sequence. Each array has
                shape ``(L, 3, 3)`` for N, CA, C atoms.
            sequences: Native amino acid sequences corresponding to each backbone.
            n_samples: Number of sampled sequences to average over per backbone.
            temperature: Sampling temperature. Lower values produce sequences closer to
                the model's mode; higher values increase diversity.

        Returns:
            Mean sequence recovery (0-1) for each input sequence, shape ``(N,)``.
        """
        recovery = []
        for i in tqdm(
            range(0, len(sequences), self.batch_size),
            disable=not self.verbose,
        ):
            batch_coordinates = coordinates[i : i + self.batch_size]
            batch_sequences = sequences[i : i + self.batch_size]

            batch_recovery = np.zeros(len(batch_sequences))
            for _ in range(n_samples):
                generated = self.sample(batch_coordinates, temperature=temperature)
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
        confidences: list[np.ndarray] | None = None,
        temperature: float = 1.0,
    ) -> list[str]:
        """
        Sample sequences conditioned on backbone coordinates.

        Args:
            coordinates: Backbone coordinates, one array per sequence. Each array has
                shape ``(L, 3, 3)`` for N, CA, C atoms.
            confidences: Per-residue confidence scores in ``[0, 1]``, one array of shape
                ``(L,)`` per sequence. Defaults to all ones if not provided.
            temperature: Sampling temperature. Lower values produce sequences closer to
                the model's mode; higher values increase diversity.

        Returns:
            Sampled amino acid sequences, one per input backbone.
        """
        samples = []
        for i in tqdm(
            range(0, len(coordinates), self.batch_size),
            disable=not self.verbose,
        ):
            batch_coordinates = coordinates[i : i + self.batch_size]
            batch_confidences = (
                confidences[i : i + self.batch_size]
                if confidences is not None
                else None
            )

            batch_sequences = _sample(
                model=self.model,
                alphabet=self.alphabet,
                coords=batch_coordinates,
                confidences=batch_confidences,
                device=self.device,
                temperature=temperature,
                generator=self.generator,
            )

            samples.extend(batch_sequences)
        return samples


# Adapted from: https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/util.py#L125
def _tokenize(
    alphabet,
    coords: list[np.ndarray],
    sequences: list[str],
    confidences: list[np.ndarray],
    device: str,
):
    from esm.inverse_folding.util import CoordBatchConverter

    batch_converter = CoordBatchConverter(alphabet)
    coords, confidences, _, tokens, padding_mask = batch_converter.from_lists(
        coords_list=coords,
        seq_list=sequences,
        confidence_list=confidences,
        device=device,
    )
    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]
    target_mask = target != alphabet.padding_idx

    return coords, padding_mask, confidences, prev_output_tokens, target, target_mask


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
    coords, padding_mask, confidences, prev_output_tokens, targets, target_mask = (
        _tokenize(
            alphabet=alphabet,
            coords=coords,
            sequences=sequences,
            confidences=None,
            device=device,
        )
    )
    logits, _ = model.forward(coords, padding_mask, confidences, prev_output_tokens)
    perplexities = _logits_to_perplexity(logits, targets, target_mask).cpu()

    return perplexities.cpu().numpy()


# Adapted from: https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/gvp_transformer.py#L88
def _sample(
    model,
    alphabet,
    coords: list[np.ndarray],
    generator: torch.Generator,
    partial_seqs: list[str] = None,  # @TODO: test
    confidences: list[np.ndarray] = None,
    temperature: float = 1.0,
    device: str = None,
) -> list[str]:
    """
    Samples sequences based on multinomial sampling (no beam search).

    Args:
        coords: List of numpy arrays of shape: L x 3 x 3, each representing a backbone
        partial_seq: Optional, partial sequences with mask tokens if part of
            the sequence is known
        temperature: sampling temperature, use low temperature for higher
            sequence recovery and high temperature for higher diversity
        confidence: optional length L list of confidence scores for coordinates. the value is between 0 and 1.
    """

    batch_coords, padding_mask, batch_confidences, _, _, _ = _tokenize(
        alphabet=alphabet,
        coords=coords,
        sequences=None,
        confidences=confidences,
        device=device,
    )

    mask_token_id = alphabet.get_idx("<mask>")

    max_length = padding_mask.shape[-1] - 2  # excluding BOS + EOS

    sampled_tokens = torch.full(
        (len(coords), max_length + 1),
        mask_token_id,
        dtype=int,
        device=device,
    )
    sampled_tokens[:, 0] = alphabet.get_idx("<cath>")

    if partial_seqs is not None:
        for i, partial_seq in enumerate(partial_seqs):
            if partial_seq is None:
                continue
            for j, c in enumerate(partial_seq):
                sampled_tokens[i, j + 1] = alphabet.get_idx(c)

    # Save incremental states for faster sampling
    incremental_state = dict()

    # Run encoder only once
    encoder_out = model.encoder(batch_coords, padding_mask, batch_confidences)

    # Decode one token at a time
    for i in range(1, max_length + 1):
        logits, _ = model.decoder(
            sampled_tokens[:, :i],
            encoder_out,
            incremental_state=incremental_state,
        )
        logits = logits[:, :, 0]
        logits /= temperature
        probs = F.softmax(logits, dim=-1)
        sampled_tokens[:, i] = torch.multinomial(probs, 1, generator=generator).squeeze(
            -1
        )

    sampled_sequences = []

    for seq_coords, seq_tokens in zip(coords, sampled_tokens, strict=True):
        sequence = "".join(
            [alphabet.get_tok(a) for a in seq_tokens[1 : 1 + len(seq_coords)]]
        )
        sampled_sequences.append(sequence)

    return sampled_sequences


## seqme interface


def compute_perplexity(
    coordinates: list[np.ndarray],
    sequences: list[str],
    *,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Compute per-sequence perplexity of native sequences given backbone coordinates.

    Args:
        coordinates: Backbone coordinates, one array per sequence. Each array has
            shape ``(L, 3, 3)`` for N, CA, C atoms.
        sequences: Native amino acid sequences corresponding to each backbone.
        batch_size: Number of sequences per batch.
        device: Device to run inference on. Defaults to CUDA if available.

    Returns:
        Perplexity score for each sequence, shape ``(N,)``.
    """
    return ESMIF1(device=device, batch_size=batch_size).compute_perplexity(
        coordinates=coordinates,
        sequences=sequences,
    )


def compute_sequence_recovery(
    coordinates: list[np.ndarray],
    sequences: list[str],
    *,
    n_samples: int = 1,
    temperature: float = 1.0,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Estimate sequence recovery by sampling from the model and comparing against native sequences.

    Args:
        coordinates: Backbone coordinates, one array per sequence. Each array has
            shape ``(L, 3, 3)`` for N, CA, C atoms.
        sequences: Native amino acid sequences corresponding to each backbone.
        n_samples: Number of sampled sequences to average over per backbone.
        temperature: Sampling temperature. Lower values produce sequences closer to
            the model's mode; higher values increase diversity.
        batch_size: Number of sequences per batch.
        device: Device to run inference on. Defaults to CUDA if available.

    Returns:
        Mean sequence recovery (0-1) for each input sequence, shape ``(N,)``.
    """
    return ESMIF1(device=device, batch_size=batch_size).compute_sequence_recovery(
        coordinates=coordinates,
        sequences=sequences,
        n_samples=n_samples,
        temperature=temperature,
    )


def sample(
    coordinates: list[np.ndarray],
    *,
    confidence: list[np.ndarray] | None = None,
    temperature: float = 1.0,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Sample sequences conditioned on backbone coordinates.

    Args:
        coordinates: Backbone coordinates, one array per sequence. Each array has
            shape ``(L, 3, 3)`` for N, CA, C atoms.
        confidence: Per-residue confidence scores in ``[0, 1]``, one array of shape
            ``(L,)`` per sequence. Defaults to all ones if not provided.
        temperature: Sampling temperature. Lower values produce sequences closer to
            the model's mode; higher values increase diversity.
        batch_size: Number of sequences per batch.
        device: Device to run inference on. Defaults to CUDA if available.

    Returns:
        Sampled amino acid sequences, one per input backbone.
    """
    return ESMIF1(device=device, batch_size=batch_size).sample(
        coordinates=coordinates,
        confidences=confidence,
        temperature=temperature,
    )


if __name__ == "__main__":
    coords = np.ones((4, 3, 3), dtype=np.float32)
    confidence = np.ones(4, dtype=np.float32)
    sequence = "AKMM"

    n = 3
    all_coords = [coords] * n
    all_confidence = [confidence] * n
    all_sequences = [sequence] * n

    ppl = compute_perplexity(all_coords, all_sequences)
    np.testing.assert_allclose(ppl, [30.44957] * n, rtol=1e-5)

    sequences = sample(all_coords, confidence=all_confidence, temperature=1e-3)
    assert sequences == ["MGHH", "MGHH", "MGHH"]

    recovery = compute_sequence_recovery(all_coords, all_sequences)
    np.testing.assert_allclose(recovery, [0.0] * n, rtol=1e-5)
