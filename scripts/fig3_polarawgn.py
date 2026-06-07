import numpy as np
import argparse
import csv
import math

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


def add_gaussian_jamming(
    received: np.ndarray,
    noise_variance: float,
    jammer_to_noise_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """
    Continuous Gaussian jammer.
    jammer_to_noise_db is JNR = jammer_power / AWGN_noise_power in dB.
    """
    jnr = 10.0 ** (jammer_to_noise_db / 10.0)
    jammer_variance = jnr * noise_variance
    jammer = rng.normal(0.0, np.sqrt(jammer_variance), size=received.shape)
    return received + jammer, jammer_variance


def bpsk_awgn_channel(x, snr_db, N, K, jam=True, jammer_to_noise_db=10):
    R = K/N  # code rate
    s = 1 - 2 * x
    esn0_db = snr_db + 10 * np.log10(R)
    esn0 = 10 ** (esn0_db / 10)
    noise_variance = 1 / (2 * esn0)
    sigma = np.sqrt(noise_variance)
    noise = sigma * np.random.randn(len(s))
    y = s + noise
    if jam:
        y, jammer_variance = add_gaussian_jamming(
            y,
            noise_variance=noise_variance,
            jammer_to_noise_db=jammer_to_noise_db,
            rng=np.random.default_rng(),
        )
        noise_variance += jammer_variance
    # LLR for BPSK over AWGN
    llr = 2 * y / noise_variance
    return llr

def bpsk_correlated_rayleigh_channel(x, snr_db,  N, K,rho=0):
    R = K/N  # code rate
    s = 1 - 2 * x
    N = len(s)
    esn0_db = snr_db + 10 * np.log10(R)
    esn0 = 10 ** (esn0_db / 10)
    sigma = np.sqrt(1 / (2 * esn0))
    h = np.random.rayleigh(scale=1/np.sqrt(2), size=N)
    noise = sigma * np.random.randn(N)
    y = h * s + noise
    llr = 2 * h * y / (sigma ** 2)

    return llr

def f(a, b):
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

def g(a, b, c):
    return b + (1 - 2 * c) * a

def sc_decode(llr, frozen):
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    def recurse(llr_vec, offset):
        n = len(llr_vec)
        if n == 1:
            idx = offset
            if frozen[idx]:
                bit = 0
            else:
                bit = 0 if llr_vec[0] >= 0 else 1

            u_hat[idx] = bit         
            return np.array([bit])    

        half = n // 2
        left_llr = f(llr_vec[:half], llr_vec[half:])
        u_left = recurse(left_llr, offset)
        right_llr = g(llr_vec[:half], llr_vec[half:], u_left)
        u_right = recurse(right_llr, offset + half)
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





N = pow(2, 10)                
K = pow(2, 9)   

info_idx, frozen_idx = select_info_idx_ga(N, K, Q_5G)
frozen = np.zeros(N, dtype=bool)
frozen[list(frozen_idx)] = True

TARGET_FRAME_ERRORS = 200
MAX_FRAMES_PER_POINT = 100000

EBN0_DB_VALUES = [1, 2, 3, 4]
JNR_DB_VALUES = [-9, -6, -3, 0, 3]

POLAR_RESULTS_OUTPUT = "fig3_polarawgn.txt"


def polar_channel_label(jnr):
    if jnr is None:
        return "AWGN"
    return f"JNR={jnr:g} dB"


def write_polar_results(
    path,
    rows,
    target_frame_errors,
    max_frames_per_point,
):
    with open(path, "w", encoding="utf-8") as output:
        output.write("# polar simulation results\n")
        output.write("# x_axis=snr_db\n")
        output.write("# x_label=Eb/N0 (dB)\n")
        output.write("# fer_column=fer\n")
        output.write(f"# n={N}\n")
        output.write(f"# k={K}\n")
        output.write(f"# target_frame_errors={target_frame_errors}\n")
        output.write(f"# max_frames_per_point={max_frames_per_point}\n")
        output.write(
            "snr_db\tchannel\tjnr_db\tframes\tframe_errors\tbit_errors\t"
            "total_bits\tfer\tber\n"
        )

        for row in rows:
            jnr = row["jnr_db"]
            jnr_db = "nan" if jnr is None else f"{jnr:.12g}"
            output.write(
                f"{row['snr_db']:.12g}\t"
                f"{row['channel']}\t"
                f"{jnr_db}\t"
                f"{row['frames']}\t"
                f"{row['frame_errors']}\t"
                f"{row['bit_errors']}\t"
                f"{row['total_bits']}\t"
                f"{row['fer']:.12e}\t"
                f"{row['ber']:.12e}\n"
            )


def simulate_polar(target_frame_errors, max_frames_per_point):
    rows = []

    for jnr in [None] + JNR_DB_VALUES:
        label = polar_channel_label(jnr)

        for snr_db in EBN0_DB_VALUES:
            total_errors = 0
            total_bits = 0
            frame_errors = 0
            times = 0

            while frame_errors < target_frame_errors and times < max_frames_per_point:
                times += 1
                msg = np.random.randint(0, 2, K)

                x = polar_encode(msg, info_idx, N)
                if jnr is None:
                    llr = bpsk_awgn_channel(x, snr_db, N, K, jam=False)
                    # llr = bpsk_correlated_rayleigh_channel(x, snr_db, N, K, rho=0.9)
                else:
                    llr = bpsk_awgn_channel(x, snr_db, N, K, True, jnr)

                u_hat = sc_decode(llr, frozen)
                decoded_msg = u_hat[info_idx]

                bit_errors = int(np.sum(msg != decoded_msg))
                total_errors += bit_errors
                total_bits += K
                if bit_errors > 0:
                    frame_errors += 1

            ber = total_errors / total_bits if total_bits else float("nan")
            fer = frame_errors / times if times else float("nan")

            rows.append(
                {
                    "snr_db": float(snr_db),
                    "channel": "AWGN" if jnr is None else "JNR",
                    "jnr_db": jnr,
                    "frames": times,
                    "frame_errors": frame_errors,
                    "bit_errors": total_errors,
                    "total_bits": total_bits,
                    "fer": fer,
                    "ber": ber,
                }
            )

            print(
                f"{label:10s} Eb/N0={snr_db:g} dB: "
                f"FER={fer:.5g} BER={ber:.5g} ({frame_errors}/{times})"
            )

    return rows

def load_polar_results(path):
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

    if not data_lines:
        raise SystemExit(f"No data rows found in {path}")

    reader = csv.DictReader(data_lines, delimiter="\t")
    rows = []
    for row in reader:
        rows.append(
            {
                "snr_db": float(row["snr_db"]),
                "channel": row["channel"],
                "jnr_db": float(row["jnr_db"]),
                "frames": int(row["frames"]),
                "frame_errors": int(row["frame_errors"]),
                "bit_errors": int(row["bit_errors"]),
                "total_bits": int(row["total_bits"]),
                "fer": float(row["fer"]),
                "ber": float(row["ber"]),
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


def plot_polar_fer(
    data_path,
    output_path,
    show=True,
):
    rows, metadata = load_polar_results(data_path)
    import matplotlib.pyplot as plt

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    groups = grouped_rows(rows)

    plt.figure(figsize=(10, 6))
    all_plot_values = []

    for index, key in enumerate(sorted(groups, key=channel_sort_key)):
        points = sorted(groups[key], key=lambda row: row["snr_db"])
        x_values = [row["snr_db"] for row in points]
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

    plt.xlabel(metadata.get("x_label", "Eb/N0 (dB)"))
    plt.ylabel("FER")
    plt.xticks(sorted({row["snr_db"] for row in rows}))
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    if all_plot_values:
        plt.ylim(bottom=max(1e-4, min(all_plot_values) / 2), top=1.0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()

DEFAULT_DATA_FILE = POLAR_RESULTS_OUTPUT
DEFAULT_OUTPUT_FILE = "fig3_polarawgn.png"

def main():
    print(
        f"Running polar FER sweep until {TARGET_FRAME_ERRORS} errors per point. "
        f"Eb/N0={EBN0_DB_VALUES}, JNR={JNR_DB_VALUES}"
    )
    rows = simulate_polar(
        target_frame_errors=TARGET_FRAME_ERRORS,
        max_frames_per_point=MAX_FRAMES_PER_POINT,
    )
    write_polar_results(
        POLAR_RESULTS_OUTPUT,
        rows,
        target_frame_errors=TARGET_FRAME_ERRORS,
        max_frames_per_point=MAX_FRAMES_PER_POINT,
    )
    print(f"Saved polar simulation data to {POLAR_RESULTS_OUTPUT}")
    plot_polar_fer(
        data_path=DEFAULT_DATA_FILE,
        output_path=DEFAULT_OUTPUT_FILE,
        show=False,
    )


if __name__ == "__main__":
    main()
