

from __future__ import annotations

import numpy as np
import csv
import math

from ldpc import (
    LDPCCode,
    TannerGraph,
    bpsk,
    encode,
    make_ccsds_ldpc_128_64,
    make_tanner_graph,
    normalized_min_sum_decode,
    validate_code,
)

TARGET_FRAME_ERRORS = 200
MAX_FRAMES_PER_POINT = 1_000_000
MAX_ITER = 50
SEED = 7

EBN0_DB_VALUES = [1, 2, 3, 4]
JNR_DB_VALUES = [-9, -6, -3, 0, 3]

LDPC_RESULTS_OUTPUT = "fig2_ldpcfer.txt"


def noise_variance_from_ebn0(ebn0_db: float, rate: float) -> float:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * rate * ebn0)


def transmit_awgn_with_variance(
    symbols: np.ndarray,
    noise_variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(0.0, np.sqrt(noise_variance), size=symbols.shape)
    return symbols + noise


def add_gaussian_jamming(
    received: np.ndarray,
    noise_variance: float,
    jammer_to_noise_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    
    jnr = 10.0 ** (jammer_to_noise_db / 10.0)
    jammer_variance = jnr * noise_variance
    jammer = rng.normal(0.0, np.sqrt(jammer_variance), size=received.shape)
    return received + jammer, jammer_variance


def simulate_fer(
    code: LDPCCode,
    graph: TannerGraph,
    sweep_values: list[float],
    noise_variance_values: list[float],
    sweep_label: str,
    max_frames_per_point: int,
    target_frame_errors: int,
    max_iter: int,
    seed: int,
    jammer_to_noise_db: float | None = None,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    results = []

    for sweep_value, awgn_variance in zip(sweep_values, noise_variance_values):
        frame_errors = 0
        simulated = 0

        while simulated < max_frames_per_point and frame_errors < target_frame_errors:
            message = rng.integers(0, 2, size=code.k, dtype=np.uint8)
            codeword = encode(message, code.g)
            received = transmit_awgn_with_variance(
                bpsk(codeword),
                awgn_variance,
                rng,
            )
            decoder_variance = awgn_variance

            if jammer_to_noise_db is not None:
                received, jammer_variance = add_gaussian_jamming(
                    received,
                    awgn_variance,
                    jammer_to_noise_db,
                    rng,
                )
                decoder_variance = awgn_variance + jammer_variance

            decoded, converged, _ = normalized_min_sum_decode(
                graph,
                received,
                decoder_variance,
                max_iter=max_iter,
            )

            if (not converged) or np.any(decoded != codeword):
                frame_errors += 1

            simulated += 1

        fer = frame_errors / simulated
        channel_text = (
            f"JNR={jammer_to_noise_db:g} dB"
            if jammer_to_noise_db is not None
            else "AWGN"
        )
        results.append(
            {
                "sweep_value": float(sweep_value),
                "noise_variance": float(awgn_variance),
                "channel": "JNR" if jammer_to_noise_db is not None else "AWGN",
                "jnr_db": float(jammer_to_noise_db)
                if jammer_to_noise_db is not None
                else np.nan,
                "frames": simulated,
                "frame_errors": frame_errors,
                "fer": fer,
            }
        )
        if sweep_label == "Eb/N0":
            point = f"Eb/N0={sweep_value:>4.1f} dB, noiseVar={awgn_variance:.3e}"
        else:
            point = f"noiseVar={sweep_value:.3e}"
        print(
            f"{channel_text:13s} {point}: "
            f"FER={fer:.5f} ({frame_errors}/{simulated})"
        )
        if frame_errors < target_frame_errors:
            print(
                f"reached max frame cap before {target_frame_errors} errors"
            )

    return results


def write_ldpc_results(
    path: str,
    rows: list[dict[str, float | int | str]],
    sweep_label: str,
    x_label: str,
    log_x_axis: bool,
    target_frame_errors: int,
    max_frames: int,
    max_iter: int,
    seed: int,
) -> None:
    with open(path, "w", encoding="utf-8") as output:
        output.write("# ldpc_fer_awgn_jamming simulation results\n")
        output.write(f"# sweep_label={sweep_label}\n")
        output.write(f"# x_label={x_label}\n")
        output.write(f"# log_x_axis={int(log_x_axis)}\n")
        output.write("# fer_column=fer\n")
        output.write(f"# target_frame_errors={target_frame_errors}\n")
        output.write(f"# max_frames={max_frames}\n")
        output.write(f"# max_iter={max_iter}\n")
        output.write(f"# seed={seed}\n")
        output.write(
            "sweep_value\tnoise_variance\tchannel\tjnr_db\tframes\t"
            "frame_errors\tfer\n"
        )

        for row in rows:
            output.write(
                f"{row['sweep_value']:.12g}\t"
                f"{row['noise_variance']:.12e}\t"
                f"{row['channel']}\t"
                f"{row['jnr_db']:.12g}\t"
                f"{row['frames']}\t"
                f"{row['frame_errors']}\t"
                f"{row['fer']:.12e}\n"
            )

def load_tab_results(path):
    metadata = {}
    data_lines = []

    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                text = stripped[1:].strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue
            data_lines.append(line)

    reader = csv.DictReader(data_lines, delimiter="\t")
    rows = []
    for row in reader:
        rows.append(
            {
                "sweep_value": float(row["sweep_value"]),
                "noise_variance": float(row["noise_variance"]),
                "channel": row["channel"],
                "jnr_db": float(row["jnr_db"]),
                "frames": int(row["frames"]),
                "frame_errors": int(row["frame_errors"]),
                "fer": float(row["fer"]),
            }
        )

    return rows, metadata


def channel_sort_key(key):
    channel, jnr_db = key
    if channel == "AWGN" or jnr_db is None:
        return (0, 0.0)
    return (1, jnr_db)


def channel_label(key):
    channel, jnr_db = key
    if channel == "AWGN" or jnr_db is None:
        return "AWGN"
    return f"JNR={jnr_db:g} dB"


def grouped_rows(rows):
    grouped = {}
    for row in rows:
        key = (
            row["channel"],
            None if math.isnan(row["jnr_db"]) else row["jnr_db"],
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def plot_ldpc_fer(
    data_path,
    output_path,
    show,
):
    rows, metadata = load_tab_results(data_path)
    import matplotlib.pyplot as plt

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    groups = grouped_rows(rows)

    plt.figure(figsize=(7.0, 4.8))
    all_plot_values = []

    for index, key in enumerate(sorted(groups, key=channel_sort_key)):
        points = sorted(groups[key], key=lambda row: row["sweep_value"])
        x_values = [row["sweep_value"] for row in points]
        y_values = [
            max(row["fer"], 0.5 / max(row["frames"], 1))
            for row in points
        ]
        all_plot_values.extend(y_values)

        plt.semilogy(
            x_values,
            y_values,
            marker=markers[index % len(markers)],
            linewidth=2,
            label=channel_label(key),
        )

    x_label = metadata.get("x_label", "Eb/N0 (dB)")
    plt.xlabel(x_label)
    plt.ylabel("Frame Error Rate (FER)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

    if metadata.get("log_x_axis", "0") == "1":
        plt.xscale("log")
    else:
        plt.xticks(sorted({row["sweep_value"] for row in rows}))

    if all_plot_values:
        plt.ylim(bottom=max(1e-4, min(all_plot_values) / 2), top=1.0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()


def main() -> None:
    code = make_ccsds_ldpc_128_64()
    graph = make_tanner_graph(code.h)
    validate_code(code, graph, seed=SEED + 99)
    print(
        f"LDPC code: CCSDS (128,64), n={code.n}, "
        f"k={code.k}, rate={code.rate:.3f}"
    )

    sweep_values = EBN0_DB_VALUES
    noise_variance_values = [
        noise_variance_from_ebn0(ebn0_db, code.rate)
        for ebn0_db in sweep_values
    ]
    sweep_label = "Eb/N0"
    x_label = "Eb/N0 (dB)"
    log_x_axis = False

    print(
        f"Running LDPC FER sweep until {TARGET_FRAME_ERRORS} errors per point. "
        f"Eb/N0={sweep_values}, JNR={JNR_DB_VALUES}"
    )

    result_rows = []
    result_rows.extend(
        simulate_fer(
            code,
            graph,
            sweep_values,
            noise_variance_values,
            sweep_label,
            max_frames_per_point=MAX_FRAMES_PER_POINT,
            target_frame_errors=TARGET_FRAME_ERRORS,
            max_iter=MAX_ITER,
            seed=SEED + 1,
        )
    )

    for index, jnr_db in enumerate(JNR_DB_VALUES):
        result_rows.extend(
            simulate_fer(
                code,
                graph,
                sweep_values,
                noise_variance_values,
                sweep_label,
                max_frames_per_point=MAX_FRAMES_PER_POINT,
                target_frame_errors=TARGET_FRAME_ERRORS,
                max_iter=MAX_ITER,
                seed=SEED + 2 + index,
                jammer_to_noise_db=jnr_db,
            )
        )

    write_ldpc_results(
        LDPC_RESULTS_OUTPUT,
        result_rows,
        sweep_label=sweep_label,
        x_label=x_label,
        log_x_axis=log_x_axis,
        target_frame_errors=TARGET_FRAME_ERRORS,
        max_frames=MAX_FRAMES_PER_POINT,
        max_iter=MAX_ITER,
        seed=SEED,
    )
    print(f"Saved LDPC FER data to {LDPC_RESULTS_OUTPUT}")
    DEFAULT_DATA_FILE = LDPC_RESULTS_OUTPUT
    DEFAULT_OUTPUT_FILE = "fig2_ldpcfer.png"
    plot_ldpc_fer(
        data_path=DEFAULT_DATA_FILE,
        output_path=DEFAULT_OUTPUT_FILE,
        show=False,
    )



if __name__ == "__main__":
    main()
