"""用量和花费：本次用了多少 token / 合成了多少字 / 识别了多少分钟，阿里云余额和历史账单。

两件事分开：
* **本机计数**（准确、实时）：每次调用大模型记下返回的 token 数，每次合成记字数，识别记音频时长。
  按场次写进 ``usage_log.jsonl``（配置目录，只在本机），「历史消费」里能看到每一场。
* **阿里云账户**（以阿里云为准，有延迟）：用 AccessKey 调费用中心 OpenAPI 查可用余额和月度账单。
  余额差只是首次成功查询以来的账户变化，不等于本次实时费用，受出账延迟、充值、其他产品消费等影响。
  AccessKey 需要「AliyunBSSReadOnlyAccess」（只读费用）权限，存在 Windows 凭据管理器里。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

LOG_FILE = "usage_log.jsonl"
BSS_ENDPOINT = "https://business.aliyuncs.com/"
BSS_VERSION = "2017-12-14"


class UsageTracker:
    """线程安全的本场用量计数。任何一处记账失败都不能影响答辩，所以方法里不抛异常。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.started_at = time.time()
            self.llm: dict[str, list[int]] = {}      # model -> [调用次数, 输入 token, 输出 token]
            self.tts: dict[str, list[int]] = {}      # model -> [次数, 字符数]
            self.asr: dict[str, float] = {}          # model -> 秒
            self.balance_start: float | None = None
            self.balance_now: float | None = None
            self.balance_started_at: float | None = None
            self.balance_updated_at: float | None = None
            self.currency = "CNY"
            self.listeners: list = getattr(self, "listeners", [])

    def add_llm(self, model: str, input_tokens: int, output_tokens: int) -> None:
        try:
            with self._lock:
                row = self.llm.setdefault(str(model or "?"), [0, 0, 0])
                row[0] += 1
                row[1] += max(0, int(input_tokens or 0))
                row[2] += max(0, int(output_tokens or 0))
        except Exception:  # noqa: BLE001
            pass

    def add_tts(self, model: str, chars: int) -> None:
        try:
            with self._lock:
                row = self.tts.setdefault(str(model or "?"), [0, 0])
                row[0] += 1
                row[1] += max(0, int(chars or 0))
        except Exception:  # noqa: BLE001
            pass

    def add_asr(self, model: str, seconds: float) -> None:
        try:
            with self._lock:
                self.asr[str(model or "?")] = self.asr.get(str(model or "?"), 0.0) + max(0.0, float(seconds))
        except Exception:  # noqa: BLE001
            pass

    def set_balance(self, amount: float, currency: str = "CNY") -> None:
        amount = float(amount)
        if not math.isfinite(amount):
            raise ValueError("费用中心返回了无效余额")
        with self._lock:
            if self.balance_start is None or self.currency != (currency or "CNY"):
                self.balance_start = amount
                self.balance_started_at = time.time()
            self.balance_now = amount
            self.balance_updated_at = time.time()
            self.currency = currency or "CNY"

    def snapshot(self) -> dict:
        with self._lock:
            input_tokens = sum(row[1] for row in self.llm.values())
            output_tokens = sum(row[2] for row in self.llm.values())
            spent = None
            if self.balance_start is not None and self.balance_now is not None:
                spent = round(self.balance_start - self.balance_now, 4)
            return {
                "started_at": self.started_at,
                "llm": {k: list(v) for k, v in self.llm.items()},
                "tts": {k: list(v) for k, v in self.tts.items()},
                "asr": dict(self.asr),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens": input_tokens + output_tokens,
                "llm_calls": sum(row[0] for row in self.llm.values()),
                "tts_chars": sum(row[1] for row in self.tts.values()),
                "asr_seconds": round(sum(self.asr.values()), 1),
                "balance_start": self.balance_start,
                "balance_now": self.balance_now,
                "balance_started_at": self.balance_started_at,
                "balance_updated_at": self.balance_updated_at,
                "spent": spent,
                "currency": self.currency,
            }

    def empty(self) -> bool:
        snap = self.snapshot()
        return not (snap["tokens"] or snap["tts_chars"] or snap["asr_seconds"])


USAGE = UsageTracker()


def record_llm_usage(model: str, usage: dict | None, *, prompt_chars: int = 0, output_chars: int = 0) -> None:
    """从接口返回的 usage 记账；没有 usage（个别接口不返回）时按字数粗估，并不影响任何功能。"""
    try:
        if isinstance(usage, dict) and (usage.get("prompt_tokens") or usage.get("input_tokens")):
            USAGE.add_llm(model, int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                          int(usage.get("completion_tokens") or usage.get("output_tokens") or 0))
        elif prompt_chars or output_chars:
            USAGE.add_llm(model, int(prompt_chars / 1.4), int(output_chars / 1.4))
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ 按场次存档
def save_session(base_dir, tracker: UsageTracker = USAGE) -> dict | None:
    """把本场用量追加到 usage_log.jsonl。什么都没用过就不写。"""
    if tracker.empty():
        return None
    record = tracker.snapshot()
    record["ended_at"] = time.time()
    try:
        path = Path(base_dir) / LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return None
    return record


def load_history(base_dir, limit: int = 500) -> list[dict]:
    path = Path(base_dir) / LOG_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines[-limit:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    records.reverse()   # 最新的在前
    return records


# ------------------------------------------------------------------ 阿里云费用中心 OpenAPI
def _percent(value: str) -> str:
    return quote(str(value), safe="-_.~")


def sign_rpc(params: dict, secret: str, method: str = "GET") -> str:
    """阿里云 RPC 风格签名（HMAC-SHA1，SignatureVersion 1.0）。"""
    canonical = "&".join(f"{_percent(k)}={_percent(v)}" for k, v in sorted(params.items()))
    string_to_sign = f"{method}&{_percent('/')}&{_percent(canonical)}"
    digest = hmac.new(f"{secret}&".encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def bss_request(action: str, access_key_id: str, secret: str, extra: dict | None = None, *,
                proxies=None, timeout: float = 15.0, get=None) -> dict:
    """调一次费用中心接口，返回 JSON。出错抛 RuntimeError（带阿里云给的中文/英文原因）。"""
    if not access_key_id or not secret:
        raise RuntimeError("还没填阿里云 AccessKey（在「历史消费」窗口里填，需要只读费用权限）")
    params = {
        "Action": action,
        "Format": "JSON",
        "Version": BSS_VERSION,
        "AccessKeyId": access_key_id,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureVersion": "1.0",
        "SignatureNonce": uuid.uuid4().hex,
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    params.update({k: str(v) for k, v in (extra or {}).items()})
    params["Signature"] = sign_rpc(params, secret)
    if get is None:
        import requests

        get = requests.get
    response = get(BSS_ENDPOINT, params=params, timeout=timeout, proxies=proxies)
    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"费用中心返回的不是 JSON（HTTP {getattr(response, 'status_code', '?')}）") from exc
    if not getattr(response, "ok", True) or (isinstance(data, dict) and data.get("Code") not in (None, "200", "Success")
                                              and not data.get("Success", False)):
        code = data.get("Code", "") if isinstance(data, dict) else ""
        message = data.get("Message", "") if isinstance(data, dict) else ""
        hint = "（这个 AccessKey 没有费用只读权限，请在 RAM 里给它加 AliyunBSSReadOnlyAccess）" \
            if "Forbidden" in str(code) or "NoPermission" in str(code) else ""
        raise RuntimeError(f"{code} {message}{hint}".strip())
    return data


def query_balance(access_key_id: str, secret: str, **kwargs) -> tuple[float, str]:
    """(可用余额, 币种)。"""
    data = bss_request("QueryAccountBalance", access_key_id, secret, **kwargs)
    info = data.get("Data") if isinstance(data, dict) else None
    if not isinstance(info, dict) or info.get("AvailableAmount") in (None, ""):
        raise RuntimeError("费用中心未返回可用余额，请稍后重试")
    try:
        amount = float(str(info["AvailableAmount"]).replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("费用中心返回的余额格式无效") from exc
    if not math.isfinite(amount):
        raise RuntimeError("费用中心返回的余额格式无效")
    return amount, str(info.get("Currency", "CNY") or "CNY")


def balance_summary(snap: dict) -> str:
    """账户余额变化，不冒充模型用量的实时价格。"""
    if snap["balance_now"] is None:
        return "余额：尚未查询"
    symbol = "¥" if snap["currency"] == "CNY" else snap["currency"] + " "
    text = f"余额 {symbol}{snap['balance_now']:.2f}"
    change = snap.get("spent") or 0.0
    if abs(change) < 0.005:
        return text + " · 余额暂无变化（费用可能尚未出账）"
    direction = "减少" if change > 0 else "增加"
    return text + f" · 余额{direction} {symbol}{abs(change):.2f}（非实时费用）"


def query_month_bill(access_key_id: str, secret: str, cycle: str, **kwargs) -> list[dict]:
    """某月（YYYY-MM）按产品汇总的消费：[{product, amount, currency}]，金额从高到低。"""
    data = bss_request("QueryBillOverview", access_key_id, secret, {"BillingCycle": cycle}, **kwargs)
    items = ((data.get("Data") or {}).get("Items") or {}).get("Item") or []
    totals: dict[str, float] = {}
    currency = "CNY"
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("ProductName") or item.get("ProductCode") or "其他")
        detail = str(item.get("ProductDetail") or "")
        label = f"{name} · {detail}" if detail and detail != name else name
        try:
            amount = float(item.get("PretaxAmount") or item.get("PaymentAmount") or 0)
        except (TypeError, ValueError):
            amount = 0.0
        currency = str(item.get("Currency") or currency)
        totals[label] = totals.get(label, 0.0) + amount
    rows = [{"product": k, "amount": round(v, 4), "currency": currency} for k, v in totals.items() if v]
    rows.sort(key=lambda r: -r["amount"])
    return rows


def format_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 10_000:
        return f"{value / 1000:.1f}k"
    return str(value)
