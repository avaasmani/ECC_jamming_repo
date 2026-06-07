import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt

# https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/15.02.00_60/ts_138212v150200p.pdf
Q_5G = [
0,1,2,4,8,16,32,3,5,64,9,6,17,10,18,128,
12,33,65,20,256,34,24,36,7,129,66,512,11,40,68,130,
19,13,48,14,72,257,21,132,35,258,26,513,80,37,25,22,
136,260,264,38,514,96,67,41,144,28,69,42,516,49,74,272,
160,520,288,528,192,544,70,44,131,81,50,73,15,320,133,52,
23,134,384,76,137,82,56,27,97,39,259,84,138,145,261,29,
43,98,515,88,140,30,146,71,262,265,161,576,45,100,640,51,
148,46,75,266,273,517,104,162,53,193,152,77,164,768,268,274,
518,54,83,57,521,112,135,78,289,194,85,276,522,58,168,139,
99,86,60,280,89,290,529,524,196,141,101,147,176,142,530,321,
31,200,90,545,292,322,532,263,149,102,105,304,296,163,92,47,
267,385,546,324,208,386,150,153,165,106,55,328,536,577,548,113,
154,79,269,108,578,224,166,519,552,195,270,641,523,275,580,291,
59,169,560,114,277,156,87,197,116,170,61,531,525,642,281,278,
526,177,293,388,91,584,769,198,172,120,201,336,62,282,143,103,
178,294,93,644,202,592,323,392,297,770,107,180,151,209,284,648,
94,204,298,400,608,352,325,533,155,210,305,547,300,109,184,534,
537,115,167,225,326,306,772,157,656,329,110,117,212,171,776,330,
226,549,538,387,308,216,416,271,279,158,337,550,672,118,332,579,
540,389,173,121,553,199,784,179,228,338,312,704,390,174,554,581,
393,283,122,448,353,561,203,63,340,394,527,582,556,181,295,285,
232,124,205,182,643,562,286,585,299,354,211,401,185,396,344,586,
645,593,535,240,206,95,327,564,800,402,356,307,301,417,213,568,
832,588,186,646,404,227,896,594,418,302,649,771,360,539,111,331,
214,309,188,449,217,408,609,596,551,650,229,159,420,310,541,773,
610,657,333,119,600,339,218,368,652,230,391,313,450,542,334,233,
555,774,175,123,658,612,341,777,220,314,424,395,673,583,355,287,
183,234,125,557,660,616,342,316,241,778,563,345,452,397,403,207,
674,558,785,432,357,187,236,664,624,587,780,705,126,242,565,398,
346,456,358,405,303,569,244,595,189,566,676,361,706,589,215,786,
647,348,419,406,464,680,801,362,590,409,570,788,597,572,219,311,
708,598,601,651,421,792,802,611,602,410,231,688,653,248,369,190,
364,654,659,335,480,315,221,370,613,422,425,451,614,543,235,412,
343,372,775,317,222,426,453,237,559,833,804,712,834,661,808,779,
617,604,433,720,816,836,347,897,243,662,454,318,675,618,898,781,
376,428,665,736,567,840,625,238,359,457,399,787,591,678,434,677,
349,245,458,666,620,363,127,191,782,407,436,626,571,465,681,246,
707,350,599,668,790,460,249,682,573,411,803,789,709,365,440,628,
689,374,423,466,793,250,371,481,574,413,603,366,468,655,900,805,
615,684,710,429,794,252,373,605,848,690,713,632,482,806,427,904,
414,223,663,692,835,619,472,455,796,809,714,721,837,716,864,810,
606,912,722,696,377,435,817,319,621,812,484,430,838,667,488,239,
378,459,622,627,437,380,818,461,496,669,679,724,841,629,351,467,
438,737,251,462,442,441,469,247,683,842,738,899,670,783,849,820,
728,928,791,367,901,630,685,844,633,711,253,691,824,902,686,740,
850,375,444,470,483,415,485,905,795,473,634,744,852,960,865,693,
797,906,715,807,474,636,694,254,717,575,913,798,811,379,697,431,
607,489,866,723,486,908,718,813,476,856,839,725,698,914,752,868,
819,814,439,929,490,623,671,739,916,463,843,381,497,930,821,726,
961,872,492,631,729,700,443,741,845,920,382,822,851,730,498,880,
742,445,471,635,932,687,903,825,500,846,745,826,732,446,962,936,
475,853,867,637,907,487,695,746,828,753,854,857,504,799,255,964,
909,719,477,915,638,748,944,869,491,699,754,858,478,968,383,910,
815,976,870,917,727,493,873,701,931,756,860,499,731,823,922,874,
918,502,933,743,760,881,494,702,921,501,876,847,992,447,733,827,
934,882,937,963,747,505,855,924,734,829,965,938,884,506,749,945,
966,755,859,940,830,911,871,639,888,479,946,750,969,508,861,757,
970,919,875,862,758,948,977,923,972,761,877,952,495,703,935,978,
883,762,503,925,878,735,993,885,939,994,980,926,764,941,967,886,
831,947,507,889,984,751,942,996,971,890,509,949,973,1000,892,950,
863,759,1008,510,979,953,763,974,954,879,981,982,927,995,765,956,
887,985,997,986,943,891,998,766,511,988,1001,951,1002,893,975,894,
1009,955,1004,1010,957,983,958,987,1012,999,1016,767,989,1003,990,1005,
959,1018,1014,1017,991,1006,1011,1019,1007,1013,1020,1021,1022,1023
]

def polar_encode(msg_bits, info_idx, N):
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = msg_bits

    x = u.copy()
    n = int(np.log2(N))

    for stage in range(n):
        step = 2 ** (stage + 1)
        half = step // 2
        x = x.reshape(-1, step)
        x[:, :half] ^= x[:, half:]
        x = x.reshape(-1)

    return x

# ---------- BPSK + AWGN Channel ----------
def bpsk_awgn_channel(x, snr_db, N, K):
    R = K/N  # code rate
    s = 1 - 2 * x

    esn0_db = snr_db + 10 * np.log10(R)
    esn0 = 10 ** (esn0_db / 10)

    sigma = np.sqrt(1 / (2 * esn0))

    noise = sigma * np.random.randn(len(s))
    y = s + noise
    # LLR for BPSK over AWGN
    llr = 2 * y / (sigma ** 2)
    return llr

def bpsk_correlated_rayleigh_channel(x, snr_db,  N, K,rho=0):
    R = K/N  # code rate
    # BPSK modulation: 0 -> +1, 1 -> -1
    s = 1 - 2 * x

    N = len(s)
    esn0_db = snr_db + 10 * np.log10(R)
    # esn0_db = snr_db
    esn0 = 10 ** (esn0_db / 10)

    sigma = np.sqrt(1 / (2 * esn0))

    
    # snr = 10 ** (snr_db / 10)
    # sigma = np.sqrt(1 / (2 * snr))

    # Generate correlated complex Gaussian fading
    # h_real = np.zeros(N)
    # h_imag = np.zeros(N)

    # h_real[0] = np.random.randn()
    # h_imag[0] = np.random.randn()

    # for i in range(1, N):
    #     h_real[i] = rho * h_real[i-1] + np.sqrt(1 - rho**2) * np.random.randn()
    #     h_imag[i] = rho * h_imag[i-1] + np.sqrt(1 - rho**2) * np.random.randn()

    # # Rayleigh fading magnitude
    # h = np.sqrt(h_real**2 + h_imag**2) / np.sqrt(2)

    h = np.random.rayleigh(scale=1/np.sqrt(2), size=N)

    # AWGN noise
    noise = sigma * np.random.randn(N)

    # Received signal
    y = h * s + noise

    # LLR for BPSK over Rayleigh fading channel
    llr = 2 * h * y / (sigma ** 2)

    return llr

# ---------- SC Decoder ----------
def f(a, b):
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

def g(a, b, c):
    return b + (1 - 2 * c) * a

def sc_decode(llr, frozen):
    N = len(llr)
    # frozen = np.zeros(N, dtype=bool)
    # frozen[list(frozen_idx)] = True

    u_hat = np.zeros(N, dtype=int)

    def recurse(llr_vec, offset):
        n = len(llr_vec)

        if n == 1:
            idx = offset
            if frozen[idx]:
                bit = 0
            else:
                bit = 0 if llr_vec[0] >= 0 else 1

            u_hat[idx] = bit          # store real decoded u bit
            return np.array([bit])    # return partial sum for parent

        half = n // 2

        left_llr = f(llr_vec[:half], llr_vec[half:])
        u_left = recurse(left_llr, offset)

        right_llr = g(llr_vec[:half], llr_vec[half:], u_left)
        u_right = recurse(right_llr, offset + half)

        # partial sums for parent
        return np.concatenate([u_left ^ u_right, u_right])

    recurse(llr, 0)
    return u_hat

def select_info_idx_ga(N, K, Q_5G):
    Q_N = [i for i in Q_5G if i < N]
    info_idx = np.array(Q_N[-K:], dtype=int)
    frozen_idx = np.array(Q_N[:-K], dtype=int)
    info_idx = np.sort(info_idx)
    frozen_idx = np.sort(frozen_idx)

    return info_idx, frozen_idx




def run_polar_simulation() -> dict[str, object]:
    N = pow(2, 10)
    K = pow(2, 9)
    target_frame_errors = 50
    max_frames_per_point = 5000

    info_idx, frozen_idx = select_info_idx_ga(N, K, Q_5G)
    frozen = np.zeros(N, dtype=bool)
    frozen[list(frozen_idx)] = True

    fer_results = {}
    ber_results = {}
    rows = []
    snr_range = np.arange(0, 5, 1)

    for channel_type in ["AWGN", "Rayleigh"]:
        fer_list = []
        ber_list = []
        for snr_db in snr_range:
            total_errors = 0
            total_bits = 0
            frame_errors = 0
            times = 0
            while frame_errors < target_frame_errors and times < max_frames_per_point:
                times += 1
                msg = np.random.randint(0, 2, K)

                x = polar_encode(msg, info_idx, N)
                if channel_type == "AWGN":
                    llr = bpsk_awgn_channel(x, snr_db, N, K)
                else:
                    llr = bpsk_correlated_rayleigh_channel(x, snr_db, N, K, rho=0.9)

                u_hat = sc_decode(llr, frozen)
                decoded_msg = u_hat[info_idx]

                bit_errors = int(np.sum(msg != decoded_msg))
                total_errors += bit_errors
                total_bits += K
                if bit_errors > 0:
                    frame_errors += 1

            ber = total_errors / total_bits
            fer = max(frame_errors / times, 1e-5)
            row = {
                "snr_db": int(snr_db),
                "encoding_scheme": "polar",
                "channel": channel_type,
                "frames": times,
                "frame_errors": frame_errors,
                "bit_errors": total_errors,
                "total_bits": total_bits,
                "fer": fer,
                "ber": ber,
            }

            ber_list.append(ber)
            fer_list.append(fer)
            rows.append(row)
            print(f"Polar {channel_type}: Eb/N0={snr_db}, FER={fer:.4e}, BER={ber:.4e}")

        fer_results[channel_type] = fer_list
        ber_results[channel_type] = ber_list

    return {
        "x_axis": "snr_db",
        "x_label": "Eb/N0 (dB)",
        "fer_column": "fer",
        "n": N,
        "k": K,
        "target_frame_errors": target_frame_errors,
        "max_frames_per_point": max_frames_per_point,
        "snr_range": snr_range.tolist(),
        "fer_results": fer_results,
        "ber_results": ber_results,
        "rows": rows,
    }


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
        seed_chunks = [hex_word_to_bits_msb(word, circulant_size) for word in hex_words]

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


def ldpc_encode(message: np.ndarray, g: np.ndarray) -> np.ndarray:
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


def transmit_rayleigh(symbols: np.ndarray, ebn0_db: float, rate: float, rng: np.random.Generator):
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    noise_variance = 1.0 / (2.0 * rate * ebn0)

    channel_gain = rng.rayleigh(scale=1.0 / np.sqrt(2.0), size=symbols.shape)
    noise = rng.normal(0.0, np.sqrt(noise_variance), size=symbols.shape)

    received = channel_gain * symbols + noise
    return received, channel_gain, noise_variance


def make_tanner_graph(h: np.ndarray) -> TannerGraph:
    rows, cols = h.shape
    check_nodes, variable_nodes = np.where(h)
    return TannerGraph(
        h=h,
        check_nodes=check_nodes,
        variable_nodes=variable_nodes,
        checks_for_var=[np.where(variable_nodes == variable)[0] for variable in range(cols)],
        vars_for_check=[np.where(check_nodes == check)[0] for check in range(rows)],
    )


def normalized_min_sum_decode(
    graph: TannerGraph,
    received: np.ndarray,
    noise_variance: float,
    channel_gain: np.ndarray | None = None,
    max_iter: int = 50,
    llr_clip: float = 30.0,
    alpha: float = 0.8,
) -> tuple[np.ndarray, bool, int]:
    if channel_gain is None:
        channel_llr = np.clip(2.0 * received / noise_variance, -llr_clip, llr_clip)
    else:
        channel_llr = np.clip(2.0 * received * channel_gain / noise_variance, -llr_clip, llr_clip)

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
            incoming = np.sum(check_to_var[graph.checks_for_var[var]]) - check_to_var[edge_id]
            var_to_check[edge_id] = np.clip(channel_llr[var] + incoming, -llr_clip, llr_clip)

    return decoded, False, max_iter


def simulate_ldpc_point(
    code: LDPCCode,
    graph: TannerGraph,
    ebn0_db: float,
    target_frame_errors: int,
    max_frames: int,
    max_iter: int,
    rng: np.random.Generator,
    channel: str = "awgn",
) -> dict[str, float | int]:
    frames = 0
    frame_errors = 0
    bit_errors = 0
    raw_bit_errors = 0
    failed_convergence = 0
    total_iterations = 0

    while frames < max_frames and frame_errors < target_frame_errors:
        message = rng.integers(0, 2, size=code.k, dtype=np.uint8)
        codeword = ldpc_encode(message, code.g)

        if channel == "awgn":
            received, noise_variance = transmit_awgn(bpsk(codeword), ebn0_db, code.rate, rng)
            channel_gain = None
            hard_metric = received
        elif channel == "rayleigh":
            received, channel_gain, noise_variance = transmit_rayleigh(
                bpsk(codeword),
                ebn0_db,
                code.rate,
                rng,
            )
            hard_metric = channel_gain * received
        else:
            raise ValueError(f"Unknown channel: {channel}")

        hard_bits = (hard_metric < 0).astype(np.uint8)
        raw_bit_errors += int(np.sum(hard_bits != codeword))

        decoded_codeword, converged, iterations = normalized_min_sum_decode(
            graph,
            received,
            noise_variance,
            channel_gain=channel_gain,
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
        "snr_db": ebn0_db,
        "frames": frames,
        "frame_errors": frame_errors,
        "fer": frame_errors / frames,
        "bit_errors": bit_errors,
        "total_bits": frames * code.k,
        "ber": bit_errors / (frames * code.k),
        "raw_bit_errors": raw_bit_errors,
        "raw_ber": raw_bit_errors / (frames * code.n),
        "failed_convergence": failed_convergence,
        "avg_iterations": total_iterations / frames,
    }


def run_ldpc_simulation() -> dict[str, object]:
    code = make_ccsds_ldpc_128_64()
    graph = make_tanner_graph(code.h)

    ebn0_values = [0, 1, 2, 3, 4]
    target_frame_errors = 50
    max_frames = 5000
    max_iter = 50

    awgn_rng = np.random.default_rng(0)
    rayleigh_rng = np.random.default_rng(0)

    awgn_results = [
        simulate_ldpc_point(
            code,
            graph,
            ebn0_db,
            target_frame_errors,
            max_frames,
            max_iter,
            awgn_rng,
            channel="awgn",
        )
        for ebn0_db in ebn0_values
    ]

    rayleigh_results = [
        simulate_ldpc_point(
            code,
            graph,
            ebn0_db,
            target_frame_errors,
            max_frames,
            max_iter,
            rayleigh_rng,
            channel="rayleigh",
        )
        for ebn0_db in ebn0_values
    ]

    print("LDPC AWGN results:")
    for result in awgn_results:
        print(
            f"Eb/N0={result['ebn0_db']:.1f} dB, "
            f"FER={result['fer']:.4e} "
            f"({result['frame_errors']}/{result['frames']}), "
            f"BER={result['ber']:.4e}"
        )

    print("LDPC Rayleigh results:")
    for result in rayleigh_results:
        print(
            f"Eb/N0={result['ebn0_db']:.1f} dB, "
            f"FER={result['fer']:.4e} "
            f"({result['frame_errors']}/{result['frames']}), "
            f"BER={result['ber']:.4e}"
        )

    rows = []
    for channel_type, results in (("AWGN", awgn_results), ("Rayleigh", rayleigh_results)):
        for result in results:
            rows.append(
                {
                    "snr_db": int(result["snr_db"]),
                    "encoding_scheme": "ldpc",
                    "channel": channel_type,
                    "frames": result["frames"],
                    "frame_errors": result["frame_errors"],
                    "bit_errors": result["bit_errors"],
                    "total_bits": result["total_bits"],
                    "fer": result["fer"],
                    "ber": result["ber"],
                }
            )

    return {
        "x_axis": "snr_db",
        "x_label": "Eb/N0 (dB)",
        "fer_column": "fer",
        "n": code.n,
        "k": code.k,
        "target_frame_errors": target_frame_errors,
        "max_frames_per_point": max_frames,
        "snr_range": ebn0_values,
        "awgn_results": awgn_results,
        "rayleigh_results": rayleigh_results,
        "rows": rows,
    }


def rows_for(rows: list[dict[str, object]], encoding_scheme: str, channel: str) -> list[dict[str, object]]:
    return [
        row for row in rows
        if row["encoding_scheme"] == encoding_scheme and row["channel"] == channel
    ]


def format_results_table(title: str, results: dict[str, object]) -> str:
    lines = [
        f"# {title} simulation results",
        f"# x_axis={results['x_axis']}",
        f"# x_label={results['x_label']}",
        f"# fer_column={results['fer_column']}",
        f"# target_frame_errors={results['target_frame_errors']}",
        f"# max_frames_per_point={results['max_frames_per_point']}",
        "snr_db\tencoding_scheme\tchannel\tframes\tframe_errors\tbit_errors\ttotal_bits\tfer\tber",
    ]
    for row in results["rows"]:
        lines.append(
            f"{row['snr_db']}\t"
            f"{row['encoding_scheme']}\t"
            f"{row['channel']}\t"
            f"{row['frames']}\t"
            f"{row['frame_errors']}\t"
            f"{row['bit_errors']}\t"
            f"{row['total_bits']}\t"
            f"{row['fer']:.12e}\t"
            f"{row['ber']:.12e}"
        )
    return "\n".join(lines)


def print_results_table(title: str, results: dict[str, object]) -> None:
    print(format_results_table(title, results))
    print()


def save_results_tables(polar_results: dict[str, object], ldpc_results: dict[str, object]) -> None:
    output_path = Path(__file__).with_name("awgn_vs_rayleigh.txt")
    output_text = "\n\n".join(
        [
            format_results_table("polar", polar_results),
            format_results_table("ldpc", ldpc_results),
        ]
    )
    output_path.write_text(output_text + "\n", encoding="utf-8")
    print(f"Saved simulation tables to {output_path}")


def plot_combined_results(polar_results: dict[str, object], ldpc_results: dict[str, object]) -> None:
    polar_rows = polar_results["rows"]
    ldpc_rows = ldpc_results["rows"]
    ebn0_values = ldpc_results["snr_range"]
    ldpc_awgn_rows = rows_for(ldpc_rows, "ldpc", "AWGN")
    ldpc_rayleigh_rows = rows_for(ldpc_rows, "ldpc", "Rayleigh")

    plt.figure()
    plt.semilogy(
        [r["snr_db"] for r in ldpc_awgn_rows],
        [r["fer"] for r in ldpc_awgn_rows],
        "o--",
        label="LDPC AWGN",
    )
    plt.semilogy(
        [r["snr_db"] for r in ldpc_rayleigh_rows],
        [r["fer"] for r in ldpc_rayleigh_rows],
        "s--",
        label="LDPC Rayleigh",
    )

    polar_markers = ["^", "D", "v", "P", "X", "*"]
    for channel_type, marker in zip(["AWGN", "Rayleigh"], polar_markers):
        channel_rows = rows_for(polar_rows, "polar", channel_type)
        plt.semilogy(
            [r["snr_db"] for r in channel_rows],
            [r["fer"] for r in channel_rows],
            linestyle="-",
            marker=marker,
            linewidth=2,
            label=f"Polar {channel_type}",
        )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("FER")
    plt.ylim(1e-4, 1)
    plt.xticks(ebn0_values)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    figure_path = Path(__file__).with_name("fig1_awgn_vs_rayleigh.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined FER figure to {figure_path}")
    plt.show()


def main() -> None:
    polar_results = run_polar_simulation()
    ldpc_results = run_ldpc_simulation()
    print_results_table("polar", polar_results)
    print_results_table("ldpc", ldpc_results)
    save_results_tables(polar_results, ldpc_results)
    plot_combined_results(polar_results, ldpc_results)


if __name__ == "__main__":
    main()
