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
if args.k == "loss":
    # [HH:MM:SS] ... loss: 0.123 / 1.2e-4
    pattern = re.compile(
        r"\[(\d+):(\d+):(\d+)\].*?\bloss:\s*([0-9eE\.\+-]+)"
    )
elif args.k == "rouge1":
    # 'rouge1': 0.123 | "rouge1":0.123 | rouge1=0.123
    pattern = re.compile(
        r"(?:['\"]?rouge1['\"]?\s*[:=]\s*)([0-9]*\.?[0-9]+)"
    )

# ===== 时间状态（用于跨天）=====
base_date = None
current_date = None
last_time_sec = None

# ===== 解析 =====
with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for idx, line in enumerate(f):
        m = pattern.search(line)
        if not m:
            continue

        # ---------- LOSS ----------
        if args.k == "loss":
            h, m_, s = map(int, m.groups()[:3])
            cur_sec = h * 3600 + m_ * 60 + s

            if base_date is None:
                # 第一条时间，作为基准日期
                base_date = datetime(2000, 1, 1)
                current_date = base_date
            else:
                # 时间回跳 -> 跨天
                if cur_sec < last_time_sec:
                    current_date += timedelta(days=1)

            last_time_sec = cur_sec

            x = current_date.replace(hour=h, minute=m_, second=s)
            y = float(m.group(4))

        # ---------- ROUGE1 ----------
        else:
            # 如果日志中也有 [HH:MM:SS]，可以共用时间轴
            time_match = re.search(r"\[(\d+):(\d+):(\d+)\]", line)
            if time_match:
                h, m_, s = map(int, time_match.groups())
                cur_sec = h * 3600 + m_ * 60 + s

                if base_date is None:
                    base_date = datetime(2000, 1, 1)
                    current_date = base_date
                else:
                    if cur_sec < last_time_sec:
                        current_date += timedelta(days=1)

                last_time_sec = cur_sec
                x = current_date.replace(hour=h, minute=m_, second=s)
            else:
                # fallback：用行号（极少出现）
                x = idx

            y = float(m.group(1))

        xs.append(x)
        ys.append(y)

print(f"Parsed {len(ys)} points")

# ===== 绘图 =====
plt.figure(figsize=(8, 5))
plt.plot(xs, ys, linewidth=1.5)
plt.grid(True)

plt.xlabel("Time")
plt.ylabel(args.k.upper())
plt.title(f"{args.k.upper()} Curve")

plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(out_file, dpi=300)
plt.close()

print(f"Saved figure to {out_file}")
