"""
Parse scaling study logs (new format with η_max) into data dicts.
Handles partial logs — only returns completed (nqubit, H) blocks.
"""

import re

def parse_log(path):
    """
    Returns: dict { nqubit: { H_actual: [(nstate, eta_max), ...] } }
    Only includes blocks with a completed "→ η peak" summary line.
    """
    data = {}
    current_n = None
    current_H = None
    current_pts = []
    completed = False

    with open(path) as f:
        for line in f:
            # New nqubit block
            m = re.search(r"nqubit=(\d+)\s+base_nstate=(\d+)", line)
            if m:
                # Save last H block of the previous nqubit section
                if current_n is not None and current_H is not None and completed and current_pts:
                    data[current_n][current_H] = sorted(current_pts)
                current_n = int(m.group(1))
                if current_n not in data:
                    data[current_n] = {}
                current_H = None
                current_pts = []
                completed = False
                continue

            # New H block
            m = re.search(r"^H=([\d.]+)\s+\(k=\d+,\s*H_act=([\d.]+)\)", line.strip())
            if m:
                # Save previous block if completed
                if current_H is not None and completed and current_pts:
                    data[current_n][current_H] = sorted(current_pts)
                current_H = float(m.group(2))
                current_pts = []
                completed = False
                continue

            # Data line
            m = re.search(r"nstate=(\d+)\s+acc_med=[\d.]+\s+η_med=[\d.]+\s+η_max=([\d.]+)", line)
            if m:
                current_pts.append((int(m.group(1)), float(m.group(2))))
                continue

            # Peak summary line — marks block as complete
            m = re.search(r"→ η_max peak at nstate=(\d+):", line)
            if m:
                completed = True
                continue

    # Save last block
    if current_n is not None and current_H is not None and completed and current_pts:
        data[current_n][current_H] = sorted(current_pts)

    return data


def merge_data(*dicts):
    """Merge multiple parsed data dicts, later dicts overwrite earlier ones."""
    merged = {}
    for d in dicts:
        for n, h_data in d.items():
            if n not in merged:
                merged[n] = {}
            for H, pts in h_data.items():
                merged[n][H] = pts
    return merged


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        d = parse_log(path)
        for n in sorted(d):
            for H in sorted(d[n]):
                best = max(d[n][H], key=lambda x: x[1])
                print(f"  n={n}  H={H:.4f}  peak nstate={best[0]}  η_max={best[1]:.4f}  ({len(d[n][H])} points)")
