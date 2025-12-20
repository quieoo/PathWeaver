import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def file_match_all_keywords(filename: str, keywords: list[str]) -> bool:
    """判断文件名是否同时包含所有关键字"""
    return all(k in filename for k in keywords)


def vis_v2(
    attn_dir: str,
    keywords: list[str] = [],
    output_dir: str = "./attn_heatmaps",
    kb_len: int | None = None,
    show_kb_only: bool = True,
    vmax: float | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有符合条件的 .npy 文件
    all_files = []
    for root, _, files in os.walk(attn_dir):
        for f in files:
            if f.endswith(".npy") and file_match_all_keywords(f, keywords):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"[WARN] 未找到同时包含 {keywords} 的注意力文件")
        return

    print(f"[INFO] 共找到 {len(all_files)} 个文件")
    for path in sorted(all_files):
        try:
            attn = np.load(path)
            attn_mean = attn.mean(axis=(0, 1))  # 平均所有head

            query_len, total_len = attn_mean.shape
            if kb_len is None:
                kb_len = total_len

            kb_attn = attn_mean[:, :kb_len]
            kb_token_importance = kb_attn.max(axis=0)

            plt.plot(range(kb_len), kb_token_importance)
            title = os.path.basename(path).replace(".npy", "")
            plt.title(f"{title}")
            plt.xlabel("KB Token Index")
            plt.ylabel("Max Attention")
            plt.savefig(os.path.join(output_dir, f"{title}.png"))
            plt.close()
        except Exception as e:
            print(f"[ERROR] 处理 {path} 时出错: {e}")

    print(f"[DONE] 所有热力图已保存至 {output_dir}")


def visualize_attention_maps(
    attn_dir: str,
    keywords: list[str] = [],
    output_dir: str = "./attn_heatmaps",
    kb_len: int | None = None,
    show_kb_only: bool = True,
    vmax: float | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    all_files = []
    for root, _, files in os.walk(attn_dir):
        for f in files:
            if f.endswith(".npy") and file_match_all_keywords(f, keywords):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"[WARN] 未找到同时包含 {keywords} 的注意力文件")
        return

    print(f"[INFO] 共找到 {len(all_files)} 个文件")

    for path in sorted(all_files):
        try:
            attn = np.load(path)
            attn_mean = attn.mean(axis=(0, 1))

            if show_kb_only and kb_len is not None:
                attn_to_plot = attn_mean[:, :kb_len]
            else:
                attn_to_plot = attn_mean

            plt.figure(figsize=(10, 6))
            plt.imshow(attn_to_plot, cmap="hot", aspect="auto", vmax=vmax)
            plt.colorbar(label="Attention Weight")
            plt.xlabel("Key / KB Token Index")
            plt.ylabel("Query Token Index")

            title = os.path.basename(path).replace(".npy", "")
            plt.title(f"Attention Heatmap - {title}")

            save_path = os.path.join(output_dir, f"{title}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"[OK] 已保存热力图：{save_path}")

        except Exception as e:
            print(f"[ERROR] 处理 {path} 时出错: {e}")

    print(f"[DONE] 所有热力图已保存至 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_dir", type=str, required=True)
    parser.add_argument(
        "--keywords", nargs="*", default=[], help="支持多个关键字，例如 --keywords key1 key2"
    )
    parser.add_argument("--output_dir", type=str, default="./attn_heatmaps")
    parser.add_argument("--kb_len", type=int, default=None)
    parser.add_argument("--show_kb_only", action="store_true")
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--show", type=str, default="average")
    args = parser.parse_args()

    if args.show == "average":
        visualize_attention_maps(
            attn_dir=args.attn_dir,
            keywords=args.keywords,
            output_dir=args.output_dir,
            kb_len=args.kb_len,
            show_kb_only=args.show_kb_only,
            vmax=args.vmax,
        )
    elif args.show == "max":
        vis_v2(
            attn_dir=args.attn_dir,
            keywords=args.keywords,
            output_dir=args.output_dir,
            kb_len=args.kb_len,
            show_kb_only=args.show_kb_only,
            vmax=args.vmax,
        )
