import time
import numpy as np
import argparse
import csv
import math
from remoteRF.drivers.adalm_pluto import *


from ldpc import (
    make_ccsds_ldpc_128_64,
    make_tanner_graph,
    normalized_min_sum_decode,
)



# SDR parameters

sample_rate = 1e6

tx_carrier_freq_Hz = 915e6
rx_carrier_freq_Hz = 915e6

tx_rf_bw_Hz = sample_rate
rx_rf_bw_Hz = sample_rate

tx_gain_dB_values = [-46,-44,-42,-40,-38]

rx_gain_dB = 40
rx_agc_mode = "manual"

rx_buffer_size = 50_000
tx_cyclic_buffer = True

tx_scale = 2**13
tx_settle_s = 0.2
rx_clear_reads = 1

target_ldpc_errors_per_point = 50

max_valid_frames_per_gain = 10000
max_capture_attempts_per_gain = 50000
progress_print_interval = 100
verbose_output = False

remove_rx_dc = True

RX_TOKEN = "BD7NOCVPURw"
TX_TOKEN = "rUHwL8fNRNs"

# CFO correction

use_tone_cfo_correction = True
tone_freq_hz = 250e3
tone_search_half_width_hz = 100e3
tone_dc_exclusion_hz = 20e3
tone_samples = 20_000
tone_scale = 2**13
tone_tx_gain_dB = -10.0

# Jamming sweep parameters

jammer_to_noise_db_values = [None, -9, -3, 3]

jammer_seed = 2
jammer_duty_cycle = 1.0

sdr_results_output = "sdr_results.txt"

# Frame / modem / LDPC parameters

sps = 10

Nzc = 127
zc_root = 25

rrc_beta = 0.5
rrc_span = 9

guard_symbols = 500

payload_bits_per_block = 64
coded_bits_per_block = 128
ldpc_max_iters = 50

sync_candidate_count = 4
sync_search_radius = sps
payload_extra_shift_radius = sps
preamble_candidate_limit = 3
min_corr_norm = 0.1

np.random.seed(1)

# Header parameters

HEADER_FRAME_NUMBER = 17
HEADER_MODULATION_ID = 0          # 00 = QPSK
HEADER_CODE_ID = 0                # 00 = LDPC(128,64)

HEADER_NUM_LDPC_BLOCKS = 1
HEADER_PAYLOAD_LENGTH_BITS = 64

HEADER_RESERVED = 0

HEADER_INFO_BITS_WITHOUT_CRC = 40
HEADER_CRC_BITS = 8
HEADER_INFO_BITS = HEADER_INFO_BITS_WITHOUT_CRC + HEADER_CRC_BITS

POLAR_HEADER_N = 128
POLAR_HEADER_K = HEADER_INFO_BITS

# Basic helpers

def log(message=""):
    if verbose_output:
        print(message, flush=True)


def status(message=""):
    print(message, flush=True)


def zadoff_chu_seq(root, length):
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)


def qpsk_mod(bits):
    bits = np.asarray(bits).astype(int)
    b0 = bits[0::2]
    b1 = bits[1::2]
    i = 1 - 2 * b1
    q = 1 - 2 * b0
    return (i + 1j * q) / np.sqrt(2)


def qpsk_demod(symbols):
    symbols = np.asarray(symbols)
    bits = np.empty(2 * len(symbols), dtype=int)
    bits[0::2] = (np.imag(symbols) < 0).astype(int)
    bits[1::2] = (np.real(symbols) < 0).astype(int)
    return bits


def qpsk_llrs(symbols, noise_variance):
    symbols = np.asarray(symbols)
    noise_variance = max(float(noise_variance), 1e-5)
    amplitude = 1 / np.sqrt(2)
    llrs = np.empty(2 * len(symbols), dtype=float)
    llrs[0::2] = (2 * amplitude * np.imag(symbols)) / noise_variance
    llrs[1::2] = (2 * amplitude * np.real(symbols)) / noise_variance
    return llrs


def create_pulse_train(symbols):
    pulse_train = np.zeros(len(symbols) * sps, dtype=np.complex64)
    pulse_train[::sps] = symbols
    return pulse_train


def get_rrc_pulse(beta, span, samples_per_symbol):
    t = (
        np.arange(-span * samples_per_symbol, span * samples_per_symbol + 1)
        / samples_per_symbol
    )
    pulse = np.zeros_like(t, dtype=float)

    for index, ti in enumerate(t):
        if abs(ti) < 1e-12:
            pulse[index] = 1 + beta * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-12:
            pulse[index] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            pulse[index] = numerator / denominator

    return pulse / np.sqrt(np.sum(pulse**2))


def pulse_shape_symbols(symbols, pulse):
    return np.convolve(create_pulse_train(symbols), pulse)


def apply_frequency_correction(signal, cfo_hz):
    if abs(cfo_hz) < 1e-9:
        return signal

    n = np.arange(len(signal))
    return signal * np.exp(-1j * 2 * np.pi * cfo_hz * n / sample_rate)


def build_tone():
    t = np.arange(tone_samples) / sample_rate
    tone = 0.5 * np.exp(2j * np.pi * tone_freq_hz * t)
    return tone / (np.max(np.abs(tone)) + 1e-12) * tone_scale


def estimate_tone_cfo(sdr_tx, sdr_rx):
    tx_signal = build_tone()

    sdr_tx.tx_hardwaregain_chan0 = float(tone_tx_gain_dB)
    sdr_tx.tx_destroy_buffer()
    sdr_tx.tx(tx_signal.astype(np.complex64))
    time.sleep(tx_settle_s)
    sdr_rx.rx_destroy_buffer()
    for _ in range(rx_clear_reads):
        _ = sdr_rx.rx()

    rx_signal = sdr_rx.rx().astype(np.complex64)

    if remove_rx_dc:
        rx_signal = rx_signal - np.mean(rx_signal)

    window = np.hanning(len(rx_signal))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(rx_signal * window)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(rx_signal), d=1 / sample_rate))
    search_mask = (
        (np.abs(freqs - tone_freq_hz) <= tone_search_half_width_hz)
        & (np.abs(freqs) >= tone_dc_exclusion_hz)
    )
    if not np.any(search_mask):
        search_mask = np.abs(freqs) >= tone_dc_exclusion_hz

    candidate_indices = np.flatnonzero(search_mask)
    peak_idx = int(candidate_indices[np.argmax(spectrum[candidate_indices])])
    peak_freq_hz = float(freqs[peak_idx])
    cfo_hz = peak_freq_hz - tone_freq_hz
    peak_mag = float(spectrum[peak_idx])
    floor_mag = float(np.median(spectrum[search_mask]) + 1e-12)
    peak_to_floor_db = 20 * np.log10(peak_mag / floor_mag)
    status(
        "Tone CFO check: "
        f"expected={tone_freq_hz / 1e3:.1f} kHz, "
        f"measured={peak_freq_hz / 1e3:.1f} kHz, "
        f"coarse_CFO={cfo_hz:.1f} Hz, "
        f"peak/floor={peak_to_floor_db:.1f} dB"
    )
    return float(cfo_hz)


def estimated_symbol_snr_db(noise_variance):
    return 10.0 * np.log10(1.0 / (2.0 * max(float(noise_variance), 1e-12)))

# Header CRC-8 + Polar helpers

def int_to_bits(value, width):
    return np.array(
        [(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)],
        dtype=np.uint8,
    )


def bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def crc8_bits(data_bits, poly=0x07, init=0x00):
    crc = init
    for bit in np.asarray(data_bits).astype(np.uint8):
        crc ^= int(bit) << 7
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return int_to_bits(crc, 8)


def pack_header_bits(
    frame_number,
    modulation_id,
    code_id,
    num_ldpc_blocks,
    payload_length_bits,
    reserved,
):
    header_wo_crc = np.concatenate(
        [
            int_to_bits(frame_number, 8),
            int_to_bits(modulation_id, 2),
            int_to_bits(code_id, 2),
            int_to_bits(num_ldpc_blocks, 8),
            int_to_bits(payload_length_bits, 16),
            int_to_bits(reserved, 4),
        ]
    ).astype(np.uint8)
    crc_bits = crc8_bits(header_wo_crc)
    header_bits = np.concatenate([header_wo_crc, crc_bits]).astype(np.uint8)
    return header_bits


def unpack_header_bits(header_bits):
    header_bits = np.asarray(header_bits).astype(np.uint8)
    header_wo_crc = header_bits[:HEADER_INFO_BITS_WITHOUT_CRC]
    received_crc = header_bits[HEADER_INFO_BITS_WITHOUT_CRC:]
    computed_crc = crc8_bits(header_wo_crc)
    crc_ok = bool(np.array_equal(received_crc, computed_crc))
    index = 0
    frame_number = bits_to_int(header_bits[index:index + 8])
    index += 8
    modulation_id = bits_to_int(header_bits[index:index + 2])
    index += 2
    code_id = bits_to_int(header_bits[index:index + 2])
    index += 2
    num_ldpc_blocks = bits_to_int(header_bits[index:index + 8])
    index += 8
    payload_length_bits = bits_to_int(header_bits[index:index + 16])
    index += 16
    reserved = bits_to_int(header_bits[index:index + 4])
    index += 4
    header_crc = bits_to_int(header_bits[index:index + 8])
    return {
        "frame_number": frame_number,
        "modulation_id": modulation_id,
        "code_id": code_id,
        "num_ldpc_blocks": num_ldpc_blocks,
        "payload_length_bits": payload_length_bits,
        "reserved": reserved,
        "header_crc": header_crc,
        "crc_ok": crc_ok,
    }


def polar_reliability_order(n):
    beta = 2 ** 0.25
    m = int(np.log2(n))
    scores = []
    for i in range(n):
        score = 0.0
        for j in range(m):
            if (i >> j) & 1:
                score += beta ** j
        scores.append(score)

    return np.argsort(scores)


def polar_info_indices(n, k):
    order = polar_reliability_order(n)
    info = np.sort(order[-k:])
    return info.astype(int)


POLAR_HEADER_INFO_INDICES = polar_info_indices(POLAR_HEADER_N, POLAR_HEADER_K)

POLAR_HEADER_FROZEN_MASK = np.ones(POLAR_HEADER_N, dtype=bool)
POLAR_HEADER_FROZEN_MASK[POLAR_HEADER_INFO_INDICES] = False


def polar_transform(u):
    x = np.asarray(u).astype(np.uint8).copy()
    n = len(x)
    step = 1

    while step < n:
        for start in range(0, n, 2 * step):
            left = slice(start, start + step)
            right = slice(start + step, start + 2 * step)
            x[left] ^= x[right]
        step *= 2

    return x


def polar_encode(info_bits, n=POLAR_HEADER_N, info_indices=POLAR_HEADER_INFO_INDICES):
    info_bits = np.asarray(info_bits).astype(np.uint8)

    u = np.zeros(n, dtype=np.uint8)
    u[info_indices] = info_bits

    return polar_transform(u)


def polar_sc_decode(llrs, frozen_mask):
    llrs = np.asarray(llrs, dtype=float)
    frozen_mask = np.asarray(frozen_mask, dtype=bool)

    def f_func(a, b):
        return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def g_func(a, b, u):
        return b + (1 - 2 * u) * a

    def recurse(local_llrs, local_frozen):
        n = len(local_llrs)

        if n == 1:
            if local_frozen[0]:
                u_hat = np.array([0], dtype=np.uint8)
            else:
                u_hat = np.array([1 if local_llrs[0] < 0 else 0], dtype=np.uint8)

            x_hat = u_hat.copy()
            return u_hat, x_hat

        half = n // 2

        left_llrs = f_func(local_llrs[:half], local_llrs[half:])
        u_left, x_left = recurse(left_llrs, local_frozen[:half])

        right_llrs = g_func(local_llrs[:half], local_llrs[half:], x_left)
        u_right, x_right = recurse(right_llrs, local_frozen[half:])

        u_hat = np.concatenate([u_left, u_right])
        x_hat = np.concatenate([x_left ^ x_right, x_right]).astype(np.uint8)

        return u_hat, x_hat

    u_hat, _ = recurse(llrs, frozen_mask)

    return u_hat


def encode_header_to_symbols():
    header_bits = pack_header_bits(
        frame_number=HEADER_FRAME_NUMBER,
        modulation_id=HEADER_MODULATION_ID,
        code_id=HEADER_CODE_ID,
        num_ldpc_blocks=HEADER_NUM_LDPC_BLOCKS,
        payload_length_bits=HEADER_PAYLOAD_LENGTH_BITS,
        reserved=HEADER_RESERVED,
    )

    polar_codeword = polar_encode(header_bits)
    header_symbols = qpsk_mod(polar_codeword)

    return {
        "header_bits": header_bits,
        "polar_codeword": polar_codeword,
        "header_symbols": header_symbols,
    }



def build_header_candidate_codebook():
    """
    Build a small codebook of valid polar-coded headers.

    Currently varies only frame_number from 0 to 255 and keeps:
        modulation_id       fixed
        code_id             fixed
        num_ldpc_blocks     fixed
        payload_length_bits fixed
        reserved            fixed

    This is appropriate for the current experiment and avoids SC decoder
    bit-ordering issues.
    """

    candidates = []

    for frame_number in range(256):
        header_bits = pack_header_bits(
            frame_number=frame_number,
            modulation_id=HEADER_MODULATION_ID,
            code_id=HEADER_CODE_ID,
            num_ldpc_blocks=HEADER_NUM_LDPC_BLOCKS,
            payload_length_bits=HEADER_PAYLOAD_LENGTH_BITS,
            reserved=HEADER_RESERVED,
        )

        polar_codeword = polar_encode(header_bits)

        candidates.append(
            {
                "frame_number": frame_number,
                "header_bits": header_bits,
                "polar_codeword": polar_codeword,
            }
        )

    return candidates


HEADER_CANDIDATE_CODEBOOK = build_header_candidate_codebook()


def decode_header_from_symbols(rx_header_symbols, noise_variance):
    """
    Decode the polar-coded header using ML search over valid header candidates.

    Positive LLR means bit 0 is more likely.
    """

    header_llrs = qpsk_llrs(rx_header_symbols, noise_variance)

    best_candidate = None
    best_metric = -np.inf

    for candidate in HEADER_CANDIDATE_CODEBOOK:
        cw = candidate["polar_codeword"].astype(np.int8)

        signs = 1 - 2 * cw
        metric = float(np.sum(signs * header_llrs))

        if metric > best_metric:
            best_metric = metric
            best_candidate = candidate

    header_bits_hat = best_candidate["header_bits"].astype(np.uint8)

    parsed = unpack_header_bits(header_bits_hat)
    parsed["header_bits"] = header_bits_hat
    parsed["ml_metric"] = best_metric

    return parsed

def polar_header_self_test(trials=10):
    """
    Noiseless test for the polar-coded header with ML header decoding.
    """

    failures = 0

    for _ in range(trials):
        header = encode_header_to_symbols()

        # Noiseless QPSK symbols
        rx_header_symbols = header["header_symbols"].copy()

        decoded = decode_header_from_symbols(
            rx_header_symbols,
            noise_variance=1e-3,
        )

        if (
            not decoded["crc_ok"]
            or decoded["frame_number"] != HEADER_FRAME_NUMBER
            or decoded["modulation_id"] != HEADER_MODULATION_ID
            or decoded["code_id"] != HEADER_CODE_ID
            or decoded["num_ldpc_blocks"] != HEADER_NUM_LDPC_BLOCKS
            or decoded["payload_length_bits"] != HEADER_PAYLOAD_LENGTH_BITS
            or decoded["reserved"] != HEADER_RESERVED
        ):
            failures += 1

            log("\nPolar header failure")
            log("Decoded header:")
            log(decoded)
            log("Expected:")
            log(
                {
                    "frame_number": HEADER_FRAME_NUMBER,
                    "modulation_id": HEADER_MODULATION_ID,
                    "code_id": HEADER_CODE_ID,
                    "num_ldpc_blocks": HEADER_NUM_LDPC_BLOCKS,
                    "payload_length_bits": HEADER_PAYLOAD_LENGTH_BITS,
                    "reserved": HEADER_RESERVED,
                }
            )

            break

    log(f"Polar header noiseless self-test failures: {failures}/{trials}")

    return failures == 0


def add_gaussian_jamming(
    received,
    noise_variance,
    jammer_to_noise_db,
    rng,
    duty_cycle=1.0,
):
    total_complex_noise_power = 2.0 * noise_variance

    jnr_linear = 10.0 ** (jammer_to_noise_db / 10.0)
    total_complex_jammer_power = jnr_linear * total_complex_noise_power

    std = np.sqrt(total_complex_jammer_power / 2.0)

    jammer = (
        rng.normal(0.0, std, size=received.shape)
        + 1j * rng.normal(0.0, std, size=received.shape)
    )

    if duty_cycle >= 1.0:
        return received + jammer, total_complex_jammer_power

    active = rng.random(received.shape) < duty_cycle
    return received + active * jammer, total_complex_jammer_power


def jammer_label(jammer_to_noise_db):
    if jammer_to_noise_db is None:
        return "No jamming"
    return f"JNR={jammer_to_noise_db:g} dB"


def progress_jammer_label(jammer_to_noise_db):
    if jammer_to_noise_db is None:
        return "No"
    return f"{jammer_to_noise_db:g}dB"


def print_sdr_progress(
    tx_gain_dB,
    iteration,
    total_valid_frames,
    stats_by_jammer,
):
    fer_values = []

    for jammer_to_noise_db in jammer_to_noise_db_values:
        stats = stats_by_jammer[jammer_to_noise_db]
        ldpc_fer = stats["ldpc_errors"] / max(stats["ldpc_blocks"], 1)
        fer_values.append(
            f"{progress_jammer_label(jammer_to_noise_db)}="
            f"{ldpc_fer:.4e}({stats['ldpc_errors']}/{stats['ldpc_blocks']})"
        )

    print(
        f"TX gain={tx_gain_dB:6.1f} dB "
        f"iter={iteration:05d} "
        f"valid_frames={total_valid_frames:05d} "
        "LDPC_FER["
        + " ".join(fer_values)
        + "]",
        flush=True,
    )


def write_sdr_results(path, rows):
    with open(path, "w", encoding="utf-8") as output:
        output.write("# sdrcodepolar SDR sweep results\n")
        output.write("# x_axis=tx_gain_dB\n")
        output.write("# fer_column=ldpc_fer\n")
        output.write("# ber_column=raw_ber\n")
        output.write(
            "tx_gain_dB\tchannel\tjnr_db\tframes\tldpc_blocks\tinvalid\t"
            "attempts\tldpc_errors\tldpc_fer\traw_ber\traw_bits\theader_fail_rate\t"
            "nonconv_rate\tavg_iters\tdone\n"
        )

        for row in rows:
            jammer_to_noise_db = row["jammer_to_noise_db"]
            channel = "AWGN" if jammer_to_noise_db is None else "JNR"
            jnr_db = (
                "nan"
                if jammer_to_noise_db is None
                else f"{jammer_to_noise_db:.12g}"
            )

            output.write(
                f"{row['tx_gain_dB']:.12g}\t"
                f"{channel}\t"
                f"{jnr_db}\t"
                f"{row['frames']}\t"
                f"{row['ldpc_blocks']}\t"
                f"{row['invalid']}\t"
                f"{row['attempts']}\t"
                f"{row['ldpc_errors']}\t"
                f"{row['ldpc_fer']:.12e}\t"
                f"{row['raw_ber']:.12e}\t"
                f"{row['raw_bits']}\t"
                f"{row['header_fail_rate']:.12e}\t"
                f"{row['nonconv_rate']:.12e}\t"
                f"{row['avg_iters']:.12g}\t"
                f"{row['done']}\n"
            )



def make_noiseless_llrs_from_codeword(codeword, llr_magnitude=20.0):
    codeword = np.asarray(codeword).astype(int)
    return np.where(codeword == 0, llr_magnitude, -llr_magnitude)


def ldpc_noiseless_self_test(code, graph, trials=100):
    failures = 0

    for trial in range(trials):
        message = np.random.randint(
            0,
            2,
            size=payload_bits_per_block,
            dtype=np.uint8,
        )

        codeword = (message @ code.g % 2).astype(np.uint8)
        llrs = make_noiseless_llrs_from_codeword(codeword)

        decoded_cw, converged, iterations = normalized_min_sum_decode(
            graph,
            llrs,
            noise_variance=1.0,
            max_iter=ldpc_max_iters,
        )

        bit_errors = int(np.sum(decoded_cw != codeword))

        if (not converged) or bit_errors > 0:
            failures += 1
            log("\nLDPC noiseless failure")
            log(f"trial: {trial}")
            log(f"converged: {converged}")
            log(f"iterations: {iterations}")
            log(f"bit errors: {bit_errors}")
            break

    log(f"LDPC noiseless self-test failures: {failures}/{trials}")
    return failures == 0


def build_frame_waveform(code, zc_seq, pulse):
    """
    Builds:

        ZC preamble | Polar-coded header | 1 LDPC payload block | guard
    """

    header = encode_header_to_symbols()

    payload_bits = np.random.randint(
        0,
        2,
        size=HEADER_PAYLOAD_LENGTH_BITS,
        dtype=np.uint8,
    )

    total_ldpc_info_bits = HEADER_NUM_LDPC_BLOCKS * payload_bits_per_block


    padded_payload_bits = np.zeros(total_ldpc_info_bits, dtype=np.uint8)
    padded_payload_bits[:HEADER_PAYLOAD_LENGTH_BITS] = payload_bits

    messages = padded_payload_bits.reshape(
        HEADER_NUM_LDPC_BLOCKS,
        payload_bits_per_block,
    )

    codewords = (messages @ code.g % 2).astype(np.uint8)

    ldpc_coded_bits = codewords.reshape(-1)
    ldpc_payload_symbols = qpsk_mod(ldpc_coded_bits)

    data_symbols = np.concatenate(
        [
            header["header_symbols"],
            ldpc_payload_symbols,
        ]
    )

    packet_symbols = np.concatenate(
        [
            zc_seq,
            data_symbols,
            np.zeros(guard_symbols, dtype=np.complex64),
        ]
    )

    tx_samples = pulse_shape_symbols(packet_symbols, pulse)
    tx_samples = tx_samples / (np.max(np.abs(tx_samples)) + 1e-12)

    return {
        "header_bits": header["header_bits"],
        "header_polar_codeword": header["polar_codeword"],
        "header_symbols": header["header_symbols"],
        "payload_bits": payload_bits,
        "padded_payload_bits": padded_payload_bits,
        "messages": messages,
        "codewords": codewords,
        "ldpc_payload_symbols": ldpc_payload_symbols,
        "data_symbols": data_symbols,
        "tx_samples": tx_samples,
    }



def estimate_preamble(rx_preamble, reference_preamble):
    mixed = rx_preamble * np.conj(reference_preamble)
    phase = np.unwrap(np.angle(mixed))
    symbol_index = np.arange(len(phase))

    slope, intercept = np.polyfit(symbol_index, phase, 1)

    preamble_corrected = rx_preamble * np.exp(
        -1j * (slope * symbol_index + intercept)
    )

    h_est = np.vdot(reference_preamble, preamble_corrected) / (
        np.vdot(reference_preamble, reference_preamble) + 1e-12
    )

    if np.abs(h_est) < 1e-12:
        return None

    preamble_eq = preamble_corrected / h_est
    preamble_evm = float(np.mean(np.abs(preamble_eq - reference_preamble) ** 2))
    residual_cfo_hz = slope * (sample_rate / sps) / (2 * np.pi)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "h_est": h_est,
        "preamble_evm": preamble_evm,
        "noise_variance": max(preamble_evm / 2.0, 1e-5),
        "residual_cfo_hz": float(residual_cfo_hz),
    }


def extract_symbols(rx_filtered, start, count):
    stop = start + count * sps

    if start < 0 or stop > len(rx_filtered):
        return None

    symbols = rx_filtered[start:stop:sps]

    if len(symbols) != count:
        return None

    return symbols


def build_preamble_candidates(rx_filtered, corr, pulse, pulse_shaped_zc, zc_seq):
    top_count = min(sync_candidate_count, len(corr))
    top_peaks = np.argpartition(corr, -top_count)[-top_count:]

    group_delay = len(pulse) // 2
    preamble_len = len(zc_seq)

    candidates = []

    for peak_index in top_peaks:
        for sync_shift in range(-sync_search_radius, sync_search_radius + 1):
            base_index = int(peak_index) + sync_shift

            for timing_offset in range(sps):
                preamble_start = base_index + group_delay + timing_offset

                rx_preamble = extract_symbols(
                    rx_filtered,
                    preamble_start,
                    preamble_len,
                )

                if rx_preamble is None:
                    continue

                preamble = estimate_preamble(rx_preamble, zc_seq)

                if preamble is None:
                    continue

                segment = rx_filtered[
                    int(peak_index):int(peak_index) + len(pulse_shaped_zc)
                ]

                corr_norm = float(
                    corr[int(peak_index)]
                    / (
                        np.linalg.norm(segment) * np.linalg.norm(pulse_shaped_zc)
                        + 1e-12
                    )
                )

                if corr_norm < min_corr_norm:
                    continue

                candidates.append(
                    {
                        "peak_index": int(peak_index),
                        "preamble_start": preamble_start,
                        "preamble_stop": preamble_start + preamble_len * sps,
                        "sync_shift": sync_shift,
                        "timing_offset": timing_offset,
                        "corr_norm": corr_norm,
                        **preamble,
                    }
                )

    candidates.sort(key=lambda item: item["preamble_evm"])
    return candidates[:preamble_candidate_limit]


def equalize_payload(rx_payload, candidate, payload_start, payload_count):
    payload_index = (
        (payload_start - candidate["preamble_start"]) / sps
        + np.arange(payload_count)
    )

    phase = candidate["slope"] * payload_index + candidate["intercept"]
    rx_payload_corrected = rx_payload * np.exp(-1j * phase)

    return rx_payload_corrected / candidate["h_est"]


def capture_and_equalize_once(
    sdr_rx,
    data_symbols,
    zc_seq,
    pulse,
    pulse_shaped_zc,
    coarse_cfo_hz,
):
    sdr_rx.rx_destroy_buffer()

    for _ in range(rx_clear_reads):
        _ = sdr_rx.rx()

    rx_signal = sdr_rx.rx().astype(np.complex64)

    if remove_rx_dc:
        rx_signal = rx_signal - np.mean(rx_signal)

    rx_signal = apply_frequency_correction(rx_signal, coarse_cfo_hz)

    rx_rms = float(np.sqrt(np.mean(np.abs(rx_signal) ** 2)))
    rx_peak = float(np.max(np.abs(rx_signal)))

    rx_filtered = np.convolve(rx_signal, pulse[::-1].conj(), mode="same")
    corr = np.abs(np.correlate(rx_filtered, pulse_shaped_zc, mode="valid"))

    preamble_candidates = build_preamble_candidates(
        rx_filtered,
        corr,
        pulse,
        pulse_shaped_zc,
        zc_seq,
    )

    if not preamble_candidates:
        return {
            "valid": False,
            "reason": "no sync candidate",
            "rx_rms": rx_rms,
            "rx_peak": rx_peak,
            "max_corr": float(np.max(corr)),
        }

    data_symbol_count = len(data_symbols)
    best = None

    for candidate in preamble_candidates:
        for payload_extra_shift in range(
            -payload_extra_shift_radius,
            payload_extra_shift_radius + 1,
        ):
            data_start = candidate["preamble_stop"] + payload_extra_shift

            rx_data = extract_symbols(
                rx_filtered,
                data_start,
                data_symbol_count,
            )

            if rx_data is None:
                continue

            rx_symbols = equalize_payload(
                rx_data,
                candidate,
                data_start,
                data_symbol_count,
            )

            known_evm = float(np.mean(np.abs(rx_symbols - data_symbols) ** 2))
            payload_noise_variance = max(
                candidate["noise_variance"],
                known_evm / 2.0,
                1e-5,
            )

            hard_bits = qpsk_demod(rx_symbols)
            remod_symbols = qpsk_mod(hard_bits)
            blind_evm = float(np.mean(np.abs(rx_symbols - remod_symbols) ** 2))

            score = (
                known_evm,
                blind_evm,
                candidate["preamble_evm"],
            )

            if best is None or score < best["score"]:
                best = {
                    "valid": True,
                    **candidate,
                    "score": score,
                    "rx_symbols": rx_symbols,
                    "payload_evm": known_evm,
                    "blind_payload_evm": blind_evm,
                    "payload_extra_shift": payload_extra_shift,
                    "preamble_noise_variance": candidate["noise_variance"],
                    "noise_variance": payload_noise_variance,
                    "channel_snr_db": estimated_symbol_snr_db(
                        payload_noise_variance
                    ),
                    "rx_rms": rx_rms,
                    "rx_peak": rx_peak,
                    "coarse_cfo_hz": coarse_cfo_hz,
                }

    if best is None:
        return {
            "valid": False,
            "reason": "no payload candidate",
            "rx_rms": rx_rms,
            "rx_peak": rx_peak,
            "max_corr": float(np.max(corr)),
        }

    return best



def decode_ldpc_payload_blocks(graph, codewords, rx_payload_symbols, noise_variance):
    num_blocks = codewords.shape[0]
    symbols_per_block = coded_bits_per_block // 2

    total_raw_bit_errors = 0
    total_decoded_bit_errors = 0
    total_block_errors = 0
    total_nonconverged = 0
    total_iterations = 0

    for block_index in range(num_blocks):
        sym_start = block_index * symbols_per_block
        sym_stop = sym_start + symbols_per_block

        rx_symbols_block = rx_payload_symbols[sym_start:sym_stop]
        codeword = codewords[block_index]

        hard_bits = qpsk_demod(rx_symbols_block)
        raw_bit_errors = int(np.sum(hard_bits != codeword))

        llrs = qpsk_llrs(rx_symbols_block, noise_variance)

        decoded_cw, converged, iterations = normalized_min_sum_decode(
            graph,
            llrs,
            noise_variance=1.0,
            max_iter=ldpc_max_iters,
        )

        decoded_bit_errors = int(np.sum(decoded_cw != codeword))
        block_error = int((not converged) or decoded_bit_errors > 0)

        total_raw_bit_errors += raw_bit_errors
        total_decoded_bit_errors += decoded_bit_errors
        total_block_errors += block_error
        total_nonconverged += int(not converged)
        total_iterations += iterations

    total_coded_bits = num_blocks * coded_bits_per_block

    return {
        "num_blocks": num_blocks,
        "raw_bit_errors": total_raw_bit_errors,
        "raw_bits": total_coded_bits,
        "raw_ber": total_raw_bit_errors / total_coded_bits,
        "ldpc_block_errors": total_block_errors,
        "ldpc_block_fer": total_block_errors / num_blocks,
        "nonconverged": total_nonconverged,
        "avg_iterations": total_iterations / num_blocks,
        "decoded_bit_errors": total_decoded_bit_errors,
    }


def decode_frame_symbols(
    graph,
    codewords,
    rx_data_symbols,
    noise_variance,
):
    header_symbol_count = POLAR_HEADER_N // 2
    ldpc_payload_symbol_count = HEADER_NUM_LDPC_BLOCKS * (coded_bits_per_block // 2)

    rx_header_symbols = rx_data_symbols[:header_symbol_count]
    rx_ldpc_payload_symbols = rx_data_symbols[
        header_symbol_count:header_symbol_count + ldpc_payload_symbol_count
    ]

    decoded_header = decode_header_from_symbols(
        rx_header_symbols,
        noise_variance,
    )

    payload_decode = decode_ldpc_payload_blocks(
        graph,
        codewords,
        rx_ldpc_payload_symbols,
        noise_variance,
    )

    return {
        "header": decoded_header,
        "payload": payload_decode,
    }


def decode_frame_with_jammer(
    graph,
    codewords,
    rx_data_symbols,
    channel_noise_variance,
    jammer_to_noise_db,
    rng,
):
    if jammer_to_noise_db is None:
        jammed_symbols = rx_data_symbols
        total_complex_jammer_power = 0.0
        total_llr_noise_variance = channel_noise_variance
    else:
        jammed_symbols, total_complex_jammer_power = add_gaussian_jamming(
            rx_data_symbols.copy(),
            channel_noise_variance,
            jammer_to_noise_db,
            rng,
            duty_cycle=jammer_duty_cycle,
        )

        jammer_variance_per_real_dim = total_complex_jammer_power / 2.0
        total_llr_noise_variance = channel_noise_variance + jammer_variance_per_real_dim

    decoded = decode_frame_symbols(
        graph,
        codewords,
        jammed_symbols,
        total_llr_noise_variance,
    )

    decoded["jammer_power_complex"] = total_complex_jammer_power
    decoded["total_llr_noise_variance"] = total_llr_noise_variance
    decoded["jammer_to_noise_db"] = jammer_to_noise_db

    return decoded


def load_sdr_results(path):
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
                "tx_gain_dB": float(row["tx_gain_dB"]),
                "channel": row["channel"],
                "jnr_db": float(row["jnr_db"]),
                "frames": int(row["frames"]),
                "ldpc_blocks": int(row["ldpc_blocks"]),
                "invalid": int(row["invalid"]),
                "attempts": int(row["attempts"]),
                "ldpc_errors": int(row["ldpc_errors"]),
                "ldpc_fer": float(row["ldpc_fer"]),
                "raw_ber": float(row["raw_ber"]),
                "raw_bits": int(row["raw_bits"]),
                "header_fail_rate": float(row["header_fail_rate"]),
                "nonconv_rate": float(row["nonconv_rate"]),
                "avg_iters": float(row["avg_iters"]),
                "done": row["done"].lower() == "true",
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

def plot_sdr_fer(
    data_path,
    output_path,
    show=True,
):
    rows, _ = load_sdr_results(data_path)
    import matplotlib.pyplot as plt

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    groups = grouped_rows(rows)

    plt.figure(figsize=(7.2, 4.8))
    all_plot_values = []

    for index, key in enumerate(sorted(groups, key=channel_sort_key)):
        points = sorted(groups[key], key=lambda row: row["tx_gain_dB"])
        x_values = [row["tx_gain_dB"] for row in points]
        y_values = [
            max(row["ldpc_fer"], 0.5 / max(row["ldpc_blocks"], 1))
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

    plt.xlabel("TX gain (dB)")
    plt.ylabel("LDPC Frame Error Rate (FER)")
    plt.xticks(sorted({row["tx_gain_dB"] for row in rows}))
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

    if all_plot_values:
        plt.ylim(bottom=max(1e-4, min(all_plot_values) / 2), top=1.0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()

def plot_sdr_ber(
    data_path,
    output_path,
    show=True,
):
    rows, _ = load_sdr_results(data_path)
    import matplotlib.pyplot as plt

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    groups = grouped_rows(rows)

    plt.figure(figsize=(7.2, 4.8))
    all_plot_values = []

    for index, key in enumerate(sorted(groups, key=channel_sort_key)):
        points = sorted(groups[key], key=lambda row: row["tx_gain_dB"])
        x_values = [row["tx_gain_dB"] for row in points]
        y_values = [
            max(row["raw_ber"], 0.5 / max(row["raw_bits"], 1))
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

    plt.xlabel("TX gain (dB)")
    plt.ylabel("Raw Bit Error Rate (BER)")
    plt.xticks(sorted({row["tx_gain_dB"] for row in rows}))
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

    if all_plot_values:
        plt.ylim(bottom=max(1e-5, min(all_plot_values) / 2), top=1.0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()



def main():
    status("Starting SDR LDPC/Polar run...")
    log("Building LDPC code...")
    code = make_ccsds_ldpc_128_64()
    graph = make_tanner_graph(code.h)

    log("Running LDPC noiseless sanity check...")
    if not ldpc_noiseless_self_test(code, graph, trials=100):
        raise SystemExit("LDPC software sanity check failed. Stop before SDR.")

    log("Running polar header noiseless sanity check...")
    if not polar_header_self_test(trials=10):
        raise SystemExit("Polar header software sanity check failed. Stop before SDR.")

    log("Building ZC/RRC/header/LDPC/QPSK waveform...")
    zc_seq = zadoff_chu_seq(zc_root, Nzc)
    pulse = get_rrc_pulse(rrc_beta, rrc_span, sps)
    pulse_shaped_zc = pulse_shape_symbols(zc_seq, pulse)
    tx_data = build_frame_waveform(code, zc_seq, pulse)

    codewords = tx_data["codewords"]
    data_symbols = tx_data["data_symbols"]
    tx_signal_scaled = tx_data["tx_samples"] * tx_scale

    header_symbol_count = POLAR_HEADER_N // 2
    ldpc_payload_symbol_count = HEADER_NUM_LDPC_BLOCKS * (coded_bits_per_block // 2)

    log(
        f"Frame format: ZC({len(zc_seq)} symbols) | "
        f"Polar header({header_symbol_count} QPSK symbols) | "
        f"LDPC payload({ldpc_payload_symbol_count} QPSK symbols, "
        f"{HEADER_NUM_LDPC_BLOCKS} LDPC block) | "
        f"guard({guard_symbols} symbols)"
    )

    log(
        "Header bits: "
        f"frame_number={HEADER_FRAME_NUMBER}, "
        f"modulation_id={HEADER_MODULATION_ID:02b}, "
        f"code_id={HEADER_CODE_ID:02b}, "
        f"num_ldpc_blocks={HEADER_NUM_LDPC_BLOCKS}, "
        f"payload_length_bits={HEADER_PAYLOAD_LENGTH_BITS}, "
        f"reserved={HEADER_RESERVED:04b}"
    )

    log(
        "TX gain sweep: "
        + ", ".join(f"{value:g} dB" for value in tx_gain_dB_values)
    )

    log(f"Target LDPC block errors per TX-gain/JNR point: {target_ldpc_errors_per_point}")
    log(f"Max valid frames per TX gain: {max_valid_frames_per_gain}")
    log(f"Tone CFO correction: {use_tone_cfo_correction}")
    log(
        "JNR sweep: "
        + ", ".join(jammer_label(value) for value in jammer_to_noise_db_values)
    )

    status("Connecting SDRs...")
    log("Connecting TX SDR...")
    sdr_tx = adi.Pluto(token=TX_TOKEN)
    sdr_tx.sample_rate = int(sample_rate)

    log("Connecting RX SDR...")
    sdr_rx = adi.Pluto(token=RX_TOKEN)
    sdr_rx.sample_rate = int(sample_rate)

    log("Configuring SDRs...")
    sdr_tx.tx_destroy_buffer()
    sdr_tx.tx_rf_bandwidth = int(tx_rf_bw_Hz)
    sdr_tx.tx_lo = int(tx_carrier_freq_Hz)
    sdr_tx.tx_cyclic_buffer = tx_cyclic_buffer
    sdr_tx.tx_hardwaregain_chan0 = float(tx_gain_dB_values[0])

    sdr_rx.rx_destroy_buffer()
    sdr_rx.rx_lo = int(rx_carrier_freq_Hz)
    sdr_rx.rx_rf_bandwidth = int(rx_rf_bw_Hz)
    sdr_rx.rx_buffer_size = int(rx_buffer_size)
    sdr_rx.gain_control_mode_chan0 = rx_agc_mode
    sdr_rx.rx_hardwaregain_chan0 = float(rx_gain_dB)

    coarse_cfo_hz = 0.0

    if use_tone_cfo_correction:
        status("Running tone CFO check...")
        coarse_cfo_hz = estimate_tone_cfo(sdr_tx, sdr_rx)
    else:
        status("Tone CFO correction disabled.")

    status(f"Using coarse CFO correction: {coarse_cfo_hz:.1f} Hz")
    status("Using max(preamble noise, payload EVM/2) for LDPC/jammer variance.")

    jammer_rng = np.random.default_rng(jammer_seed)
    sweep_summary = []

    try:
        for tx_gain_dB in tx_gain_dB_values:
            log("\n" + "=" * 72)
            log(f"TX gain = {tx_gain_dB:.1f} dB")
            log("=" * 72)
            status(f"Starting TX gain {tx_gain_dB:.1f} dB")

            sdr_tx.tx_hardwaregain_chan0 = float(tx_gain_dB)
            sdr_tx.tx_destroy_buffer()
            time.sleep(0.1)
            sdr_tx.tx(tx_signal_scaled.astype(np.complex64))
            time.sleep(tx_settle_s)

            total_valid_frames = 0
            total_invalid = 0
            capture_attempts = 0

            stats_by_jammer = {
                jammer_to_noise_db: {
                    "frames": 0,
                    "ldpc_blocks": 0,
                    "raw_bit_errors": 0,
                    "raw_bits": 0,
                    "ldpc_errors": 0,
                    "header_crc_ok": 0,
                    "header_crc_fail": 0,
                    "nonconverged": 0,
                    "iterations_sum": 0.0,
                    "done": False,
                }
                for jammer_to_noise_db in jammer_to_noise_db_values
            }

            log("\nCapturing, applying synthetic jamming, and decoding...\n")

            while (
                total_valid_frames < max_valid_frames_per_gain
                and capture_attempts < max_capture_attempts_per_gain
                and not all(stats["done"] for stats in stats_by_jammer.values())
            ):
                capture_attempts += 1

                result = capture_and_equalize_once(
                    sdr_rx,
                    data_symbols,
                    zc_seq,
                    pulse,
                    pulse_shaped_zc,
                    coarse_cfo_hz,
                )

                if not result["valid"]:
                    total_invalid += 1

                    if capture_attempts % progress_print_interval == 0:
                        print_sdr_progress(
                            tx_gain_dB,
                            capture_attempts,
                            total_valid_frames,
                            stats_by_jammer,
                        )

                    continue

                total_valid_frames += 1

                for jammer_to_noise_db in jammer_to_noise_db_values:
                    stats = stats_by_jammer[jammer_to_noise_db]

                    if stats["done"]:
                        continue

                    decoded = decode_frame_with_jammer(
                        graph,
                        codewords,
                        result["rx_symbols"],
                        result["noise_variance"],
                        jammer_to_noise_db,
                        jammer_rng,
                    )

                    header = decoded["header"]
                    payload = decoded["payload"]

                    stats["frames"] += 1
                    stats["ldpc_blocks"] += payload["num_blocks"]
                    stats["raw_bit_errors"] += payload["raw_bit_errors"]
                    stats["raw_bits"] += payload["raw_bits"]
                    stats["ldpc_errors"] += payload["ldpc_block_errors"]
                    stats["nonconverged"] += payload["nonconverged"]
                    stats["iterations_sum"] += (
                        payload["avg_iterations"] * payload["num_blocks"]
                    )

                    if header["crc_ok"]:
                        stats["header_crc_ok"] += 1
                    else:
                        stats["header_crc_fail"] += 1

                    if stats["ldpc_errors"] >= target_ldpc_errors_per_point:
                        stats["done"] = True

                if capture_attempts % progress_print_interval == 0:
                    print_sdr_progress(
                        tx_gain_dB,
                        capture_attempts,
                        total_valid_frames,
                        stats_by_jammer,
                    )

            if capture_attempts % progress_print_interval != 0:
                print_sdr_progress(
                    tx_gain_dB,
                    capture_attempts,
                    total_valid_frames,
                    stats_by_jammer,
                )

            for jammer_to_noise_db in jammer_to_noise_db_values:
                label = jammer_label(jammer_to_noise_db)
                stats = stats_by_jammer[jammer_to_noise_db]

                frames = stats["frames"]
                ldpc_blocks = stats["ldpc_blocks"]
                raw_ber = stats["raw_bit_errors"] / max(stats["raw_bits"], 1)
                ldpc_fer = stats["ldpc_errors"] / max(ldpc_blocks, 1)
                header_fail_rate = stats["header_crc_fail"] / max(frames, 1)
                nonconv_rate = stats["nonconverged"] / max(ldpc_blocks, 1)
                avg_iters = stats["iterations_sum"] / max(ldpc_blocks, 1)

                sweep_summary.append(
                    {
                        "tx_gain_dB": tx_gain_dB,
                        "jammer_to_noise_db": jammer_to_noise_db,
                        "jammer_label": label,
                        "frames": frames,
                        "ldpc_blocks": ldpc_blocks,
                        "invalid": total_invalid,
                        "attempts": capture_attempts,
                        "ldpc_errors": stats["ldpc_errors"],
                        "ldpc_fer": ldpc_fer,
                        "raw_ber": raw_ber,
                        "raw_bits": stats["raw_bits"],
                        "header_fail_rate": header_fail_rate,
                        "nonconv_rate": nonconv_rate,
                        "avg_iters": avg_iters,
                        "done": stats["done"],
                    }
                )

        write_sdr_results(sdr_results_output, sweep_summary)
        FER_DATA_FILE = "sdr_results.txt"
        FER_OUTPUT_FILE = "fig5_sdrfer.png" 
        plot_sdr_fer(
            data_path=FER_DATA_FILE,
            output_path=FER_OUTPUT_FILE,
            show="true",
        )
        BER_OUTPUT_FILE = "fig6_sdrber.png"
        plot_sdr_ber(
            data_path=FER_DATA_FILE,
            output_path=BER_OUTPUT_FILE,
            show=False,
        )
    


    finally:
        log("\nStopping TX...")
        sdr_tx.tx_destroy_buffer()


if __name__ == "__main__":
    main()
