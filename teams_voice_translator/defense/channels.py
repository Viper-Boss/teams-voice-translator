"""第二通道：答辩现场的大模型请求（实时翻译、问答库匹配、生成回答）同时 / 稍后发给另一个入口，谁先出字用谁。

**只说一次**：两个通道的结果绝不会都用上。以「第一段正文」为准定胜负——哪个通道先吐出正文，
就只要它的；另一个立刻取消，它已经收到的内容直接扔掉。胜负定了以后，就算赢的那个中途断了
也不会换到另一个接着说（前半句已经说出去了，换通道等于把话再说一遍），而是按原来的方式报错。
本模块只负责大模型通道；识别与合成有独立开关和独立的故障/完整音频选路策略。

阿里云百炼的地域（2026-09 核对官方文档）：华北2（北京）、中国香港、新加坡、日本（东京）、
美国（弗吉尼亚）、德国（法兰克福）。**没有杭州、上海地域**；各地域的 API Key、业务空间、
可用模型互相独立，不能混用。所以：
* 「北京 · 公共入口」：同一个北京地域、换一个入口域名（dashscope.aliyuncs.com），用同一个 API Key，
  不用另开账号——防的是业务空间专属域名这一条线路抽风；
* 其他地域：要在那个地域开通百炼、单独建 API Key（和业务空间），模型名也可能不同（可单独指定）。
"""
from __future__ import annotations

from dataclasses import dataclass

REGIONS: tuple[tuple[str, str], ...] = (
    ("cn-beijing", "华北2（北京）"),
    ("cn-hongkong", "中国香港"),
    ("ap-southeast-1", "新加坡"),
    ("ap-northeast-1", "日本（东京）"),
    ("us-east-1", "美国（弗吉尼亚）"),
    ("eu-central-1", "德国（法兰克福）"),
)
REGION_LABELS = dict(REGIONS)

# 第二通道可选的入口。beijing-public 用主 API Key；其余要该地域自己的 API Key。
ROUTE_SAME_ENTRANCE = "primary-model"   # 入口、Key 都和主通道一样，只换模型

BACKUP_ROUTES: tuple[tuple[str, str], ...] = (
    ("beijing-public", "华北2（北京）· 公共入口 dashscope.aliyuncs.com（同一个 API Key，推荐）"),
    (ROUTE_SAME_ENTRANCE, "同一个入口 · 只换模型（同一个 API Key；比如主通道 plus、第二通道 flash）"),
    ("cn-hongkong", "中国香港（需要香港地域的 API Key）"),
    ("ap-southeast-1", "新加坡（需要新加坡地域的 API Key）"),
    ("ap-northeast-1", "日本（东京）（需要该地域的 API Key）"),
    ("us-east-1", "美国（弗吉尼亚）（需要该地域的 API Key）"),
    ("eu-central-1", "德国（法兰克福）（需要该地域的 API Key）"),
)

MODES: tuple[tuple[str, str], ...] = (
    ("race", "★ 同时请求：先出正文的一路胜出（最稳；大模型费用约翻倍，flash 很便宜）"),
    ("delayed", "主通道 1.5 秒仍无正文才请求第二通道（几乎不多花钱，卡的时候慢 1.5 秒）"),
)
DELAY_SECONDS = 1.5

PRIMARY_REGION = "cn-beijing"
COMPAT_PATH = "/compatible-mode/v1/chat/completions"
BACKUP_KEY_USERNAME = "backup_api_key"


def primary_endpoint(workspace_id: str) -> str:
    return f"https://{workspace_id}.{PRIMARY_REGION}.maas.aliyuncs.com{COMPAT_PATH}"


def uses_main_key(route: str) -> bool:
    return (route or "beijing-public") in ("beijing-public", ROUTE_SAME_ENTRANCE)


def backup_endpoint(route: str, workspace_id: str = "", primary_workspace: str = "") -> str:
    route = route or "beijing-public"
    if route == ROUTE_SAME_ENTRANCE:
        if not (primary_workspace or "").strip():
            raise ValueError("还没有填主通道的业务空间 ID")
        return primary_endpoint(primary_workspace.strip())
    if route == "beijing-public":
        return f"https://dashscope.aliyuncs.com{COMPAT_PATH}"
    workspace_id = (workspace_id or "").strip()
    if workspace_id:
        return f"https://{workspace_id}.{route}.maas.aliyuncs.com{COMPAT_PATH}"
    if route == "ap-southeast-1":
        return f"https://dashscope-intl.aliyuncs.com{COMPAT_PATH}"
    if route == "cn-hongkong":
        return f"https://cn-hongkong.dashscope.aliyuncs.com{COMPAT_PATH}"
    if route == "us-east-1":
        return f"https://dashscope-us.aliyuncs.com{COMPAT_PATH}"
    raise ValueError("此地域需要填写业务空间 ID，不能使用未公布的入口地址")


def route_label(route: str) -> str:
    return {"beijing-public": "华北2（北京）· 公共入口",
            ROUTE_SAME_ENTRANCE: "华北2（北京）· 同一个入口换模型"}.get(route, REGION_LABELS.get(route, route))


@dataclass(frozen=True)
class BackupChannel:
    endpoint: str
    api_key: str
    label: str
    model: str = ""          # 空 = 和主通道同一个模型
    mode: str = "race"       # race / delayed
    delay: float = DELAY_SECONDS


def get_backup_api_key() -> str:
    try:
        import keyring

        from ..config import KEYRING_SERVICE
        return (keyring.get_password(KEYRING_SERVICE, BACKUP_KEY_USERNAME) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def set_backup_api_key(value: str) -> None:
    import keyring

    from ..config import KEYRING_SERVICE
    value = (value or "").strip()
    if value:
        keyring.set_password(KEYRING_SERVICE, BACKUP_KEY_USERNAME, value)


def backup_channel(settings) -> BackupChannel | None:
    """按设置组装第二通道；没开、或者缺 API Key 时返回 None（等于只有主通道，行为和原来完全一样）。"""
    try:
        if not bool(settings.get("backup_channel_enabled", False)):
            return None
        route = str(settings.get("backup_route", "beijing-public") or "beijing-public")
        if uses_main_key(route):
            key = settings.get_api_key()
        else:
            key = get_backup_api_key()
        if not key:
            return None
        mode = str(settings.get("backup_channel_mode", "race") or "race")
        return BackupChannel(
            endpoint=backup_endpoint(route, str(settings.get("backup_workspace_id", "") or ""),
                                     str(getattr(settings, "workspace_id", "") or "")),
            api_key=key,
            label=route_label(route),
            model=str(settings.get("backup_model", "") or "").strip(),
            mode=mode if mode in dict(MODES) else "race",
        )
    except Exception:  # noqa: BLE001 —— 第二通道出任何问题都退回只用主通道
        return None


def channel_summary(settings) -> str:
    primary = f"第一通道：{REGION_LABELS[PRIMARY_REGION]} · 业务空间专属入口"
    if not bool(settings.get("backup_channel_enabled", False)):
        return primary + "　｜　第二通道：未开启"
    route = str(settings.get("backup_route", "beijing-public") or "beijing-public")
    channel = backup_channel(settings)
    if channel is None:
        return primary + f"　｜　第二通道：{route_label(route)}（缺 API Key，暂不生效）"
    return primary + f"　｜　第二通道：{channel.label}"


# 语音合成第二通道（只管现场发声）怎么用。和上面大模型的 MODES 分开设：合成比 flash 大模型贵，
# 而且两个模型的声音不完全一样，默认只在主通道真的没声音时才用第二通道。
TTS_MODES: tuple[tuple[str, str], ...] = (
    ("delayed", "★ 主通道 2 秒没出声 / 失败才用第二通道（声音统一，平时不多花钱）"),
    ("race", "两个同时合成，谁先合成好用谁（最快；合成费约翻倍，个别句子是第二通道的声音）"),
)
