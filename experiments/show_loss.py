import re
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import argparse
import os

# ===== 参数 =====
parser = argparse.ArgumentParser(description="Show training metric curve")
parser.add_argument("--f", type=str, required=True, help="Path to log file")
parser.add_argument("--k", type=str, default="loss", choices=["loss", "rouge1"])
args = parser.parse_args()

log_file = args.f
os.makedirs("./loss_figure", exist_ok=True)
out_file = f"./loss_figure/{os.path.basename(log_file).replace('.log','')}_{time.strftime('%Y%m%d%H%M%S')}_{args.k}.png"

xs = []
ys = []

# ===== 正则 =====
loss_pattern = re.compile(
    r"\[(\d+):(\d+):(\d+)\].*?\bloss:\s*([0-9eE\.\+-]+)"
)

rouge1_pattern = re.compile(
    r"['\"]?rouge1['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)"
)

time_pattern = re.compile(r"\[(\d+):(\d+):(\d+)\]")

# ===== 时间状态（用于跨天）=====
base_date = None
current_date = None
last_time_sec = None

# ===== 解析 =====
if args.k == "loss":
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            m = loss_pattern.search(line)
            if not m:
                continue

            h, m_, s = map(int, m.groups()[:3])
            cur_sec = h * 3600 + m_ * 60 + s

            if base_date is None:
                base_date = datetime(2000, 1, 1)
                current_date = base_date
            else:
                if last_time_sec is not None and cur_sec < last_time_sec:
                    current_date += timedelta(days=1)

            last_time_sec = cur_sec

            x = len(xs)
            y = float(m.group(4))

            xs.append(x)
            ys.append(y)

elif args.k == "rouge1":
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    matches = list(rouge1_pattern.finditer(text))
    for i, m in enumerate(matches):
        x = len(xs)
        y = float(m.group(1))
        xs.append(x)
        ys.append(y)

print(f"Parsed {len(ys)} points")

if len(ys) == 0:
    raise ValueError(f"No {args.k} values found in {log_file}")

# ===== 绘图 =====
plt.figure(figsize=(8, 5))
plt.plot(xs, ys, linewidth=1.5)
plt.grid(True)

plt.xlabel("Step")
plt.ylabel(args.k.upper())
plt.title(f"{args.k.upper()} Curve")

plt.tight_layout()
plt.savefig(out_file, dpi=300)
plt.close()

print(f"Saved figure to {out_file}")