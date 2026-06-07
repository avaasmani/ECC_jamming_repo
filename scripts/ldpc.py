from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


CCSDS_LDPC_128_64_W_HEX = (
    ("0E69", "166B", "EF4C", "0BC2"),
    ("7766", "137E", "BB24", "8418"),
    ("C480", "FEB9", "CD53", "A713"),
    ("4EAA", "22FA", "465E", "EA11"),
)

CCSDS_LDPC_128_64_H_SHIFTS = (
    ((0, 7), (2,), (14,), (6,), (), (0,), (13,), (0,)),
    ((6,), (0, 15), (0,), (1,), (0,), (), (0,), (7,)),
    ((4,), (1,), (0, 15), (14,), (11,), (0,), (), (3,)),
    ((0,), (1,), (9,), (0, 13), (14,), (1,), (0,), ()),
)


@dataclass(frozen=True)
class LDPCCode:
    h: np.ndarray
    g: np.ndarray

    @property
    def n(self) -> int:
        return self.g.shape[1]

    @property
    def k(self) -> int:
        return self.g.shape[0]

    @property
    def rate(self) -> float:
        return self.k / self.n


@dataclass(frozen=True)
class TannerGraph:
    h: np.ndarray
    check_nodes: np.ndarray
    variable_nodes: np.ndarray
    checks_for_var: list[np.ndarray]
    vars_for_check: list[np.ndarray]


def parse_db_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def hex_word_to_bits_msb(word: str, width: int = 16) -> np.ndarray:
    return np.array([int(bit) for bit in f"{int(word, 16):0{width}b}"], dtype=np.uint8)


def ccsds_right_circulant(shifts: tuple[int, ...], size: int = 16) -> np.ndarray:
    block = np.zeros((size, size), dtype=np.uint8)

    for shift in shifts:
        for row in range(size):
            block[row, (row + shift) % size] ^= 1

    return block


def make_ccsds_ldpc_128_64() -> LDPCCode:

    k = 64
    n = 128
    circulant_size = 16

    w = np.zeros((k, k), dtype=np.uint8)
    for block_row, hex_words in enumerate(CCSDS_LDPC_128_64_W_HEX):
        seed_chunks = [
            hex_word_to_bits_msb(word, circulant_size)
            for word in hex_words
        ]

        for row_shift in range(circulant_size):
            row_chunks = [np.roll(chunk, row_shift) for chunk in seed_chunks]
            w[block_row * circulant_size + row_shift] = np.concatenate(row_chunks)

    g = np.concatenate([np.eye(k, dtype=np.uint8), w], axis=1)

    h = np.zeros((n - k, n), dtype=np.uint8)
    for block_row, row_shifts in enumerate(CCSDS_LDPC_128_64_H_SHIFTS):
        for block_col, shifts in enumerate(row_shifts):
            h[
                block_row * circulant_size:(block_row + 1) * circulant_size,
                block_col * circulant_size:(block_col + 1) * circulant_size,
            ] = ccsds_right_circulant(shifts, circulant_size)

    return LDPCCode(h=h, g=g)


def encode(message: np.ndarray, g: np.ndarray) -> np.ndarray:
    return (message @ g % 2).astype(np.uint8)


def bpsk(bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * bits


def transmit_awgn(
    symbols: np.ndarray,
    ebn0_db: float,
    rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    noise_variance = 1.0 / (2.0 * rate * ebn0)
    noise = rng.normal(0.0, np.sqrt(noise_variance), size=symbols.shape)
    return symbols + noise, noise_variance


def make_tanner_graph(h: np.ndarray) -> TannerGraph:
    rows, cols = h.shape
    check_nodes, variable_nodes = np.where(h)
    return TannerGraph(
        h=h,
        check_nodes=check_nodes,
        variable_nodes=variable_nodes,
        checks_for_var=[
            np.where(variable_nodes == variable)[0]
            for variable in range(cols)
        ],
        vars_for_check=[
            np.where(check_nodes == check)[0]
            for check in range(rows)
        ],
    )


def normalized_min_sum_decode(
    graph: TannerGraph,
    received: np.ndarray,
    noise_variance: float,
    max_iter: int = 50,
    llr_clip: float = 30.0,
    alpha: float = 0.8,
) -> tuple[np.ndarray, bool, int]:
    channel_llr = np.clip(2.0 * received / noise_variance, -llr_clip, llr_clip)
    var_to_check = channel_llr[graph.variable_nodes].copy()
    check_to_var = np.zeros(len(graph.check_nodes), dtype=float)

    for iteration in range(1, max_iter + 1):
        for edge_ids in graph.vars_for_check:
            messages = var_to_check[edge_ids]
            signs = np.where(messages < 0.0, -1.0, 1.0)
            abs_messages = np.abs(messages)
            min_index = int(np.argmin(abs_messages))
            min1 = abs_messages[min_index]
            min2 = np.min(np.delete(abs_messages, min_index))
            total_sign = np.prod(signs)

            magnitudes = np.full(len(edge_ids), min1)
            magnitudes[min_index] = min2
            check_to_var[edge_ids] = alpha * total_sign * signs * magnitudes

        posterior_llr = channel_llr.copy()
        for var, edge_ids in enumerate(graph.checks_for_var):
            posterior_llr[var] += np.sum(check_to_var[edge_ids])

        decoded = (posterior_llr < 0).astype(np.uint8)
        if np.all((graph.h @ decoded) % 2 == 0):
            return decoded, True, iteration

        for edge_id, var in enumerate(graph.variable_nodes):
            incoming = (
                np.sum(check_to_var[graph.checks_for_var[var]])
                - check_to_var[edge_id]
            )
            var_to_check[edge_id] = np.clip(channel_llr[var] + incoming, -llr_clip, llr_clip)

    return decoded, False, max_iter


def validate_code(code: LDPCCode, graph: TannerGraph, seed: int) -> None:
    rng = np.random.default_rng(seed)

    for _ in range(20):
        message = rng.integers(0, 2, size=code.k, dtype=np.uint8)
        codeword = encode(message, code.g)

        decoded, converged, _ = normalized_min_sum_decode(
            graph,
            bpsk(codeword),
            noise_variance=1e-4,
            max_iter=10,
        )


def simulate_point(
    code: LDPCCode,
    graph: TannerGraph,
    ebn0_db: float,
    target_frame_errors: int,
    max_frames: int,
    max_iter: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    frames = 0
    frame_errors = 0
    bit_errors = 0
    raw_bit_errors = 0
    failed_convergence = 0
    total_iterations = 0

    while frames < max_frames and frame_errors < target_frame_errors:
        message = rng.integers(0, 2, size=code.k, dtype=np.uint8)
        codeword = encode(message, code.g)
        received, noise_variance = transmit_awgn(bpsk(codeword), ebn0_db, code.rate, rng)

        hard_bits = (received < 0).astype(np.uint8)
        raw_bit_errors += int(np.sum(hard_bits != codeword))

        decoded_codeword, converged, iterations = normalized_min_sum_decode(
            graph,
            received,
            noise_variance,
            max_iter=max_iter,
        )
        decoded_message = decoded_codeword[:code.k]
        message_errors = int(np.sum(decoded_message != message))
        codeword_errors = int(np.sum(decoded_codeword != codeword))

        if (not converged) or codeword_errors:
            frame_errors += 1
            bit_errors += message_errors
        if not converged:
            failed_convergence += 1

        total_iterations += iterations
        frames += 1

    return {
        "ebn0_db": ebn0_db,
        "frames": frames,
        "frame_errors": frame_errors,
        "fer": frame_errors / frames,
        "bit_errors": bit_errors,
        "ber": bit_errors / (frames * code.k),
        "raw_bit_errors": raw_bit_errors,
        "raw_ber": raw_bit_errors / (frames * code.n),
        "failed_convergence": failed_convergence,
        "avg_iterations": total_iterations / frames,
    }



