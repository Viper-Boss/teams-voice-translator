# 更新日志

本项目遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [1.1.0] - 2026-08-03

### 新增

- 新增 `Qwen3.5 LiveTranslate` 极速直译引擎：一次实时连接完成中文识别、英文翻译和英文语音流式播放。
- F9 默认启用极速直译，并显示松开按键至首段英文音频的实际耗时。
- 支持首次克隆、每次克隆、默认音色和固定直译 `voice_id` 四种音色策略。
- 设置页可一键创建仅供 LiveTranslate 使用的固定克隆音色。
- 保留原 ASR → Qwen-MT → Qwen-Audio-TTS 传统引擎，方便使用编辑确认、翻译记忆、领域和表达风格。

### 改进

- 极速模式自动锁定流式发送，避免“先确认”选项与已经播放的音频冲突。
- WebSocket 意外断开会立即报告原因，不再等待完整响应超时。
- API 连通性测试会根据当前引擎检查 LiveTranslate 或传统机器翻译模型。

### 验证

- 新增 LiveTranslate 会话参数、事件解析、音频流和异常断开单元测试。
- Windows 离屏界面启动测试覆盖默认极速模式与默认模型。

## [1.0.0] - 2026-08-02

首次公开发布。

### 新增

- 面向 Microsoft Teams 的中英双向语音翻译工作流。
- `F8` 原声通话与后台双语字幕，`F9` 中文转英文语音。
- Windows WASAPI 回环捕获老师/Teams 声音并生成中英字幕。
- 物理麦克风、系统声和混合会议三轨 WAV 录音。
- 可编辑确认、键盘文本发声、课程术语、翻译记忆与表达风格。
- 中英双语时间轴、歌词式悬浮字幕、搜索、崩溃恢复和课堂总结。
- SRT、TXT、WebVTT、Markdown 与 JSONL 字幕导出。
- Windows x64 便携发行包。

### 修复

- 保证连续启动的字幕缓存文件名唯一，避免 Windows 低精度系统时钟下偶发覆盖或无法恢复。

[1.1.0]: https://github.com/Viper-Boss/teams-voice-translator/releases/tag/v1.1.0
[1.0.0]: https://github.com/Viper-Boss/teams-voice-translator/releases/tag/v1.0.0
