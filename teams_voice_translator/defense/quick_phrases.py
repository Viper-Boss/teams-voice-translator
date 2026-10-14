"""快捷用语：答不上来、没听清、要争取时间时点一下，用你的克隆音色说出来。

每条都有中文（按钮上显示，给你看）和英文（真正读出来的）。默认一套放在代码里，
你改过之后存到配置目录的 ``quick_phrases.json``（不进仓库）。
英文和问答库预案走同一套合成参数，提前渲染好就能秒播、断网也能播。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PHRASES_FILE = "quick_phrases.json"

# 分类顺序即界面标签页顺序：现场最常用的放前面
CATEGORIES: list[tuple[str, str]] = [
    ("think", "争取时间"),
    ("hard", "答不上来"),
    ("clarify", "没听清"),
    ("thanks", "感谢回应"),
    ("wrap", "收尾补充"),
    ("general", "开场/设备"),
]
CATEGORY_LABELS = dict(CATEGORIES)


@dataclass
class QuickPhrase:
    id: str
    category: str
    zh: str          # 按钮上显示的中文（只看不读）
    en: str          # 朗读的英文
    star: bool = False   # 常用：会出现在助答弹窗底部，一键可点

    def label(self) -> str:
        return self.zh.strip() or self.en.strip()


def _p(pid: str, category: str, zh: str, en: str, star: bool = False) -> QuickPhrase:
    return QuickPhrase(pid, category, zh, en, star)


DEFAULT_PHRASES: list[QuickPhrase] = [
    # ---- 争取时间 ----
    _p("t1", "think", "问题很好，让我想想怎么回答",
       "Um, that's a really good question, Professor. Let me think for a moment about how best to answer it.", True),
    _p("t2", "think", "让我整理一下思路",
       "Let me just organize my thoughts for a second."),
    _p("t3", "think", "请给我几秒钟想一想",
       "Could you give me a few seconds to think about this, please?"),
    _p("t4", "think", "这个角度很有意思，我之前没这样想过",
       "That's a very interesting point. I hadn't looked at it from that angle before."),
    _p("t5", "think", "我确认一下是否理解对了您的问题",
       "Let me make sure I understand your question correctly."),
    _p("t6", "think", "我翻到相关那页 PPT",
       "Let me go back to the relevant slide for a moment."),
    _p("t7", "think", "我分两部分来回答",
       "I'd like to answer this in two parts."),
    _p("t8", "think", "这个问题涉及好几个方面",
       "That's a good question, and it involves several aspects. Let me start with the most important one."),
    # ---- 答不上来 ----
    _p("h1", "hard", "有点难，我得回去看看源码当时怎么写的",
       "Hmm, that's a tricky one. I'd need to go back and check how I implemented that part in my source code.", True),
    _p("h2", "hard", "这个问题有点难，能换个问题吗",
       "That's a difficult one for me. Would it be all right if we moved on to another question?", True),
    _p("h3", "hard", "这方面我还没深入研究过",
       "To be honest, I haven't studied that aspect in depth yet."),
    _p("h4", "hard", "我不太确定，不想给您不准确的回答",
       "I'm not completely sure about this, and I don't want to give you an inaccurate answer."),
    _p("h5", "hard", "答辩后我仔细查证再回复您",
       "I'd like to check this carefully after the defense and get back to you with a proper answer."),
    _p("h6", "hard", "这稍微超出了本文的研究范围",
       "I think that goes a bit beyond the scope of this thesis, but it's a very interesting direction."),
    _p("h7", "hard", "具体数值记不清了，在论文实验章节里",
       "I don't remember the exact value right now, but it's reported in the experiment chapter of my thesis."),
    _p("h8", "hard", "这是很好的建议，可以作为未来工作",
       "That's a great point, and it's something I plan to explore in my future work."),
    _p("h9", "hard", "这是本研究的一个局限",
       "You're right, and I think that is one of the limitations of this study."),
    # ---- 没听清 ----
    _p("c1", "clarify", "抱歉没听清，能再说一遍吗",
       "Sorry, I didn't quite catch that. Could you please repeat the question?", True),
    _p("c2", "clarify", "能说慢一点吗",
       "Sorry, could you say that a little more slowly, please?"),
    _p("c3", "clarify", "能换个说法问一下吗",
       "Could you please rephrase the question? I want to make sure I answer the right thing."),
    _p("c4", "clarify", "您指的是哪一部分",
       "Could you clarify which part of the work you're referring to?"),
    _p("c5", "clarify", "您问的是方法还是实验结果",
       "Are you asking about the method itself, or about the experimental results?"),
    _p("c6", "clarify", "网络有点卡，能再说一遍吗",
       "Sorry, the connection seems a bit unstable. Could you repeat that, please?"),
    # ---- 感谢回应 ----
    _p("k1", "thanks", "好的好的，谢谢教授",
       "Okay, okay. Thank you, Professor.", True),
    _p("k2", "thanks", "谢谢您的建议",
       "Thank you very much for the suggestion. I really appreciate it."),
    _p("k3", "thanks", "这个意见很有帮助，我会认真修改",
       "That's a very helpful comment. I will carefully revise that part of the thesis."),
    _p("k4", "thanks", "您说得对",
       "You're absolutely right, Professor."),
    _p("k5", "thanks", "我同意，这确实是不足之处",
       "I agree. That is indeed a weakness of the current work."),
    _p("k6", "thanks", "谢谢，我记下来了",
       "Thank you, I've noted that down."),
    # ---- 收尾补充 ----
    _p("w1", "wrap", "这就是我的回答，谢谢",
       "That's my answer to your question. Thank you."),
    _p("w2", "wrap", "希望这回答了您的问题",
       "I hope that answers your question."),
    _p("w3", "wrap", "需要我再详细解释哪一部分吗",
       "Would you like me to explain any part of that in more detail?"),
    _p("w4", "wrap", "我再补充一点",
       "If I may, I'd like to add one more point."),
    _p("w5", "wrap", "简单总结一下",
       "Let me briefly summarize."),
    # ---- 开场 / 设备 ----
    _p("g1", "general", "各位老师好，感谢参加",
       "Hello everyone, and thank you all for being here today."),
    _p("g2", "general", "我可以开始了吗",
       "May I start my presentation now?"),
    _p("g3", "general", "能听清我说话吗",
       "Can you hear me clearly?"),
    _p("g4", "general", "能看到我的屏幕吗",
       "Can you see my screen?"),
    _p("g5", "general", "稍等，我共享一下屏幕",
       "One moment, please. Let me share my screen."),
    _p("g6", "general", "感谢各位老师，答辩到此结束",
       "Thank you all for your time and valuable comments. That concludes my defense."),
]


def default_phrases() -> list[QuickPhrase]:
    return [QuickPhrase(**asdict(p)) for p in DEFAULT_PHRASES]


def parse_phrases(data) -> list[QuickPhrase]:
    """容错读取：缺字段的条目跳过，未知分类归到「争取时间」，重复 id 自动改名。"""
    if isinstance(data, dict):
        data = data.get("phrases", [])
    if not isinstance(data, list):
        return []
    result: list[QuickPhrase] = []
    seen: set[str] = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        en = str(item.get("en", "") or "").strip()
        zh = str(item.get("zh", "") or "").strip()
        if not en:
            continue
        category = str(item.get("category", "") or "")
        if category not in CATEGORY_LABELS:
            category = CATEGORIES[0][0]
        pid = str(item.get("id", "") or "").strip() or f"u{index}"
        while pid in seen:
            pid += "-2"
        seen.add(pid)
        result.append(QuickPhrase(pid, category, zh, en, bool(item.get("star", False))))
    return result


def next_phrase_id(phrases: list[QuickPhrase]) -> str:
    used = {p.id for p in phrases}
    n = 1
    while f"u{n}" in used:
        n += 1
    return f"u{n}"


class QuickPhraseStore:
    def __init__(self, base_dir: Path) -> None:
        self.path = Path(base_dir) / PHRASES_FILE

    def load(self) -> list[QuickPhrase]:
        """没保存过（或文件坏了）就用默认的一套。"""
        if not self.path.exists():
            return default_phrases()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default_phrases()
        phrases = parse_phrases(data)
        return phrases or default_phrases()

    def save(self, phrases: list[QuickPhrase]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"phrases": [asdict(p) for p in phrases]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def reset(self) -> list[QuickPhrase]:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return default_phrases()


# ------------------------------------------------------------------ 自己录的声音
RECORDINGS_DIR = "quick_phrase_recordings"
RECORD_RATE = 24000


def clean_recording(pcm: bytes, *, sample_rate: int = RECORD_RATE) -> bytes:
    """去掉录音首尾的静音（各留 0.15 秒），声音太小就适度放大（峰值到约 −3 dB，最多放大 4 倍）。"""
    from array import array

    if len(pcm) % 2:
        pcm = pcm[:-1]
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return b""
    frame = max(1, sample_rate // 50)          # 20 ms
    threshold = 500
    loud = [i for i in range(0, len(samples), frame)
            if max(abs(v) for v in samples[i:i + frame]) > threshold]
    if not loud:
        return samples.tobytes()
    pad = int(sample_rate * 0.15)
    start = max(0, loud[0] - pad)
    end = min(len(samples), loud[-1] + frame + pad)
    trimmed = samples[start:end]
    peak = max(abs(v) for v in trimmed) or 1
    gain = min(4.0, 23197 / peak)              # −3 dBFS
    if gain > 1.05:
        trimmed = array("h", (max(-32768, min(32767, int(v * gain))) for v in trimmed))
    return trimmed.tobytes()


class PhraseRecordings:
    """每条快捷用语可以有一段你自己录的声音（24 kHz 单声道 PCM），有就优先播它。只存本机。"""

    def __init__(self, base_dir: Path) -> None:
        self.root = Path(base_dir) / RECORDINGS_DIR

    def path_for(self, phrase_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", phrase_id)
        return self.root / f"{safe}.pcm"

    def has(self, phrase_id: str) -> bool:
        try:
            return self.path_for(phrase_id).is_file()
        except OSError:
            return False

    def load(self, phrase_id: str) -> bytes | None:
        try:
            data = self.path_for(phrase_id).read_bytes()
        except OSError:
            return None
        return data if data and len(data) % 2 == 0 else None

    def save(self, phrase_id: str, pcm: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(phrase_id)
        temp = path.with_suffix(".tmp")
        temp.write_bytes(pcm)
        temp.replace(path)

    def delete(self, phrase_id: str) -> None:
        try:
            self.path_for(phrase_id).unlink()
        except FileNotFoundError:
            pass


# ------------------------------------------------------------------ 按评委的问题推荐
SUGGEST_COUNT = 4


def build_suggest_messages(question: str, phrases: list[QuickPhrase]) -> list[dict[str, str]]:
    listing = "\n".join(f"{p.id} | {CATEGORY_LABELS.get(p.category, p.category)} | {p.zh} | {p.en}"
                        for p in phrases)
    system = (
        "你在帮一位正在论文答辩的学生。评委刚问了一个问题。学生可能一时答不上来、需要争取时间、"
        f"没听清或想礼貌回应。请从下面的快捷用语里挑出最适合这道题的 {SUGGEST_COUNT} 条，"
        "优先挑「争取时间 / 答不上来 / 没听清」里和这道题内容最贴切的（例如问实现细节就挑「要回去看源码」，"
        "问超出范围的就挑「超出研究范围」）。只输出 JSON 数组，内容是编号，例如 [\"t1\", \"h6\"]。"
    )
    user = f"评委的问题：{question.strip()}\n\n快捷用语（编号 | 分类 | 中文 | 英文）：\n{listing}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_suggestions(content: str, phrases: list[QuickPhrase]) -> list[str]:
    ids = {p.id for p in phrases}
    found: list[str] = []
    for match in re.finditer(r"[\"']?([A-Za-z][A-Za-z0-9_-]{0,15})[\"']?", content or ""):
        pid = match.group(1)
        if pid in ids and pid not in found:
            found.append(pid)
        if len(found) >= SUGGEST_COUNT:
            break
    return found
