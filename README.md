# Teams 双向课堂翻译 v1.2.2

[![Release](https://img.shields.io/github/v/release/Viper-Boss/teams-voice-translator?display_name=tag)](https://github.com/Viper-Boss/teams-voice-translator/releases/latest)
[![CI](https://github.com/Viper-Boss/teams-voice-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/Viper-Boss/teams-voice-translator/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4)](https://github.com/Viper-Boss/teams-voice-translator/releases/latest)

这是一个面向 Windows + Microsoft Teams 的本地双向课堂翻译器。程序通过阿里云百炼官方接口处理语音和文本，并使用 Windows WASAPI 回环捕获老师在 Teams 中的声音。

## 下载

- [下载最新 Windows x64 成品版](https://github.com/Viper-Boss/teams-voice-translator/releases/latest)
- [查看 v1.2.2 更新记录](CHANGELOG.md#122---2026-08-03)
- [快速开始](快速开始.txt)

发行包的 SHA-256 校验值见对应 Release 页面随附的 `.sha256` 文件。

> 本项目不附带阿里云百炼额度、API Key、VB-CABLE 驱动或 Microsoft Teams。云端模型调用会产生相应费用。

## 核心能力

- 你说中文（极速模式）：`Qwen3.5 LiveTranslate` 单条实时链路直接输出中文字幕、英文字幕和英文语音 → VB-CABLE → Teams。
- 你说中文（传统模式）：实时 ASR → Qwen-MT → Qwen-Audio-TTS，保留编辑确认、翻译记忆和表达风格能力。
- 老师说英文：捕获 Teams 扬声器声音 → 实时英文原文 → 中文字幕。
- `F8` 原声：你的声音立即进入 Teams，同时在后台生成中英文字幕。
- `F9` 翻译：默认使用极速直译，松开后流式播放英文；也可切回传统三模型模式并选择先确认和编辑。
- 键盘输入：`Enter` 翻译并发送，`Ctrl+Enter` 或 `Shift+Enter` 换行。
- 双向三轨录音：你的麦克风、老师系统声、双方混合会议各一份 WAV。
- 完整时间轴：标记“我 / 老师”、中英文、时间和处理耗时，并支持搜索和双击重新载入。
- 课程配置：按课程保存领域、术语、翻译记忆、表达风格、音色和 TTS 指令。
- 一键本地声音复刻：直接选择 iPhone `M4A`、MP3、WAV 等录音，自动转换、临时上传、创建固定音色并清理样音。
- 字幕导出：中文、英文或双语；SRT、TXT、WebVTT、Markdown、JSONL 或全部格式。
- AI 课堂总结：知识点、老师回答、作业待办、术语和待确认问题。
- 临时缓存恢复：程序异常退出后可在下次启动时恢复尚未保存的双语字幕。
- 浅色/深色主题，以及可调透明度、字号和置顶状态的歌词式悬浮字幕。

## 安装和启动

成品版直接解压并运行 `TeamsVoiceTranslator.exe`。仍需单独安装 [VB-CABLE 官方虚拟声卡](https://vb-audio.com/Cable/)，安装驱动后重启 Windows。

源码运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\run.bat
```

## 百炼配置

在“设置”中填写北京地域的 Workspace ID 和 API Key，然后测试连通性。API Key 保存到 Windows 凭据管理器，不会写入源码或 `settings.json`。

推荐在该业务空间中授权以下模型：

- `qwen3.5-livetranslate-flash-realtime`：F9 极速中文语音直译英文字幕和语音。
- `qwen3-asr-flash-realtime`：F8、老师字幕，以及极速直译的源语音转写。
- `qwen-mt-flash`、`qwen-audio-3.0-tts-flash`：传统模式、键盘翻译及发声。
- `qwen-plus`：可选的课堂总结。

极速直译首次使用请选择“服务端复刻一次（推荐，无需上传样音）”：直接按 F9 说话，服务端会从第一段语音自动复刻音色，并在本次会话内复用。百炼当前可能拒绝为 `qwen3.5-livetranslate-flash-realtime` 预创建固定音色；固定 `voice_id` 因此仅作为已有兼容音色用户的高级选项。普通 TTS 的 `voice_id` 不能直接复用。模型协议和限制见[阿里云官方文档](https://help.aliyun.com/zh/model-studio/qwen3-5-livetranslate-flash-realtime)。

## 传统 TTS 一键本地声音复刻

这套本地录音流程主要用于传统模式的 Qwen-Audio-TTS 克隆音色。LiveTranslate 推荐直接使用“服务端复刻一次”，不需要执行以下步骤。点击传统模式音色旁的“创建克隆音色”后可以直接选择本地录音，不再需要手工制作公网 URL：

1. 首次使用点击“OSS 设置”，填写私有 Bucket 所在地域、Bucket 名称和 RAM AccessKey。
2. 选择 iPhone `M4A`、MP3、WAV、AAC、CAF、FLAC、OGG、OPUS、WMA 或 MP4 音频。
3. 程序自动裁剪为最多 30 秒，并转成 24 kHz、单声道、16-bit PCM WAV。
4. 标准 WAV 使用随机对象名临时上传到私有 OSS，并生成 15 分钟 HTTPS 签名地址。
5. 百炼返回普通 TTS 的固定 `voice_id` 后，程序自动填入并保存，同时立即删除 OSS 临时样音。

阿里云建议使用 10–20 秒样音，至少包含 5 秒连续、清晰、无背景音乐的单人语音。参见[声音复刻官方指南](https://help.aliyun.com/zh/model-studio/voice-cloning-user-guide)。

建议给专用 RAM 用户只授予临时目录所需权限：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["oss:PutObject", "oss:GetObject", "oss:DeleteObject"],
      "Resource": ["acs:oss:*:*:你的Bucket/teams-voice-translator/temporary/*"]
    }
  ]
}
```

Bucket 可以保持私有。AccessKey ID、AccessKey Secret 与百炼 API Key 均保存在 Windows 凭据管理器，不写入 `settings.json`。如果临时对象删除失败，程序会明确显示对象路径，方便手工删除。建议再为 `teams-voice-translator/temporary/` 前缀设置 1 天自动删除的 OSS 生命周期规则，作为电脑断电或程序异常退出时的隐私兜底。

普通配置和日志分别位于：

```text
%APPDATA%\TeamsVoiceTranslator\settings.json
%APPDATA%\TeamsVoiceTranslator\teams-voice-translator.log
```

## 音频与 Teams 设置

程序中：

- 物理麦克风：你的真实麦克风。
- Teams 虚拟输出：`CABLE Input (VB-Audio Virtual Cable)`。
- 老师/Teams 系统声：选择与 Teams 扬声器对应的 `[Loopback]` 设备。
- 本机试听：可选；开启时请佩戴耳机。

Teams 中：

- 麦克风：`CABLE Output (VB-Audio Virtual Cable)`。
- 扬声器：你的耳机或正常播放设备。

Windows 的命名方向是正确的：程序向 `CABLE Input` 播放，Teams 从 `CABLE Output` 录音。建议使用耳机，避免扬声器声音再次进入物理麦克风。

可点击“运行音频设备诊断”检查普通输入、输出、WASAPI 回环和虚拟声卡选择。

## 双向会议操作

1. 点击“开始听老师 / Teams”，老师英文会逐句显示原文和中文翻译。
2. 按住 `F8` 可用原声回答；松开后字幕会完整进入时间轴。
3. 按住 `F9` 说中文；松开后翻译为英文并通过当前音色发给 Teams。默认极速模式会流式播放，状态栏会显示松开按键到首段英文音频的耗时。
4. `Esc` 停止当前语音，并关闭老师监听。
5. 软件自己播放英文时，会暂时阻止这段声音重新进入老师识别通道。

F9 有两种翻译引擎：

- 极速直译（推荐）：单个实时模型同时完成中文识别、英文翻译和英文语音，延迟更低；由于语音会边生成边播放，固定为自动发送。
- 传统模式：ASR、机器翻译、TTS 三段处理，延迟较高，但支持先确认、翻译记忆、领域和表达风格。

传统模式发送方式支持：

- 立即翻译并发送：延迟最低。
- 先确认/编辑：译文出现后可以修改，再点“播放/发送当前英文”。

历史记录双击后会重新载入主编辑区，便于修改和重播。

## 课程术语和配置

“表格方式编辑课程术语”比手写 JSON 更方便。术语会自动双向使用，例如：

```json
[
  {"source": "缺陷反演", "target": "defect inversion"},
  {"source": "有限元", "target": "finite element method"}
]
```

表达风格可选自然礼貌、简洁口语、正式学术或尽量逐字忠实。设置完成后在“课程配置”中保存，下次可从会议控制台快速切换。

## 三轨录音

点击“开始录音”会同时生成：

```text
microphone_时间.wav       你的物理麦克风
teacher_system_时间.wav   Teams/老师系统声
meeting_mixed_时间.wav    双方混合会议
```

三份文件都是 16 kHz、单声道、16-bit PCM WAV，适合语音归档和字幕校对。录制他人声音前请征得对方同意。

如果录音与 F8/F9 同时使用时声卡报错，请在 Windows 声音设备高级设置中关闭“允许应用程序独占控制此设备”。

## 字幕缓存和导出

录制期间，程序始终把完整的时间、来源、中文和英文写入临时 JSONL 缓存。实时显示为纯中文或纯英文不会删减缓存。

保存时可选择：

- 语言：仅中文、仅英文、中英双语。
- 格式：SRT、TXT、WebVTT、Markdown、JSONL、SRT+TXT 或全部。
- 是否显示说话人和绝对时间。

推荐同时保存 SRT + TXT：SRT 适合播放器、剪映和 Premiere；TXT 适合阅读。WebVTT 适合网页，Markdown 适合课堂笔记，JSONL 适合后续程序处理。

正常退出时若字幕尚未导出会弹出提示；异常退出留下的缓存会在下次启动时提示恢复。

## AI 课堂总结

时间轴中有内容后点击“生成课堂总结”。默认使用 `qwen-plus`，生成结果可编辑后保存为 Markdown。总结会尝试整理课程主题、知识点、老师回答、作业待办、专业术语和需要再次确认的问题。模型只应作为辅助，专有名词和截止时间仍需人工检查。

## 隐私和限制

- 开启 F8、老师字幕或传统 F9 时，对应音频会发送到百炼实时 ASR。
- 极速 F9 会把中文音频发送到 Qwen3.5 LiveTranslate，并接收英文字幕和音频流。
- 传统 F9 与双向文本会发送到 Qwen-MT；需要发声时英文会发送到 Qwen-Audio-TTS。
- 点击课堂总结时，本次双向字幕会发送到所选 Qwen 总结模型。
- API Key 由 Windows 凭据管理器保存；日志不记录 API Key。
- 使用本地声音复刻时，标准化后的短样音会临时上传到用户配置的私有 OSS，并通过短时签名 URL 提供给百炼；程序会在请求完成后尝试立即删除。
- 系统声回环会捕获所选播放设备上的声音，不只 Teams。会议期间不要在同一设备播放其他音频。
- 本软件不是医疗、法律或专业同传服务；重要内容应人工确认。

## 打包

```powershell
.\build.ps1
```

输出位于：

```text
dist\TeamsVoiceTranslator\TeamsVoiceTranslator.exe
```

## 开源许可

Copyright (C) 2026 Viper-Boss

本项目以 **GNU GPL v3.0 or later** 发布。你可以使用、研究、修改和再分发本项目；如果分发基于本项目的修改版或衍生程序，必须按 GPLv3 兼容方式同时提供对应源代码，并保留许可证与版权声明。完整条款见 [LICENSE](LICENSE)。

第三方依赖仍分别适用其原始许可证。本段仅为通俗说明，不替代许可证正文。

欢迎提交 Issue 和 Pull Request，具体见 [CONTRIBUTING.md](CONTRIBUTING.md)；安全问题请按 [SECURITY.md](SECURITY.md) 私密报告。
