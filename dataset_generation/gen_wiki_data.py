import wikipedia
import random
import json
import os
import time
from tqdm import tqdm

wikipedia.set_lang("en")  # ✅ 强制使用英文 Wikipedia
wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"

def generate_diverse_wiki_kb(num_items=987928, save_path="../datasets/wiki/wiki.json"):
    topics = [
        "artificial intelligence", "neural networks", "quantum physics", "ancient history",
        "modern literature", "genetics", "machine learning", "chemistry", "astronomy",
        "philosophy", "linguistics", "psychology", "economics", "robotics", "ecology",
        "music theory", "geology", "statistics", "cryptography", "computer vision",
        "neuroscience", "sociology", "anthropology", "climate change", "data science"
    ]

    modifiers = [
        "key concept", "overview", "principle", "phenomenon", "summary", "fact", 
        "discussion", "outline", "idea", "insight", "notion", "aspect"
    ]

    suffixes = [
        " (synthetic record)", " (auto-generated)", " (knowledge item)", 
        " (Wikipedia-derived)", " (augmented text)", "", ""
    ]

    def random_text_perturbation(text):
        """在句子基础上添加轻微改写或上下文拼接"""
        ops = [
            lambda t: t + f", which is widely studied in the field of {random.choice(topics)}.",
            lambda t: "According to studies, " + t[0].lower() + t[1:] if t else t,
            lambda t: f"In short, {t.lower()}" if len(t.split()) > 6 else t,
            lambda t: t + f" This knowledge has influenced {random.choice(topics)} research.",
            lambda t: t.replace("is", random.choice(["represents", "refers to", "describes"])),
        ]
        return random.choice(ops)(text)

    def get_random_sentence(topic):
        try:
            summary = wikipedia.summary(topic, sentences=random.randint(2, 5))
            sentences = [s.strip() for s in summary.split(".") if len(s.split()) > 4]
            if not sentences:
                return None
            sent = random.choice(sentences)
            sent = random_text_perturbation(sent)
            sent += random.choice(suffixes)
            return sent
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[DisambiguationError] Topic '{topic}' has multiple meanings: {e.options[:3]} ...")
            return None

        except wikipedia.exceptions.PageError:
            print(f"[PageError] No page found for topic: {topic}")
            return None

        except Exception as e:
            print(f"[OtherError] Topic '{topic}' -> {type(e).__name__}: {e}")
            return None

    # 断点续生
    start_index = 0
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            start_index = sum(1 for _ in f)
        print(f"🔄 已存在 {start_index} 条，将从该位置继续生成。")

    with open(save_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(start_index, num_items), desc="生成知识库"):
            topic = random.choice(topics)
            sentence = get_random_sentence(topic)
            if sentence is None:
                continue

            # key 多样化
            key_variant = f"{random.choice(modifiers).capitalize()} of {topic}: {sentence}"
            key_variant += f" #{random.randint(9999999, 10000000)}"

            entry = {
                "key_string": key_variant,
                "description": key_variant  # ✅ key 和 value 相同
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if (i + 1) % 2000 == 0:
                print(f"已生成 {i+1} 条")
                time.sleep(0.2)  # 控制 API 调用速率

    print(f"✅ 已完成生成 {num_items} 条多样化 Wikipedia 知识库，保存到 {save_path}")

if __name__ == "__main__":
    generate_diverse_wiki_kb()
