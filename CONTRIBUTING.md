# 参与开发《留子答辩器》

留子答辩器面向留学生的会议与答辩场景。欢迎修复问题、改善翻译和播放体验、补充文档；贡献内容按项目的 GPL-3.0-or-later 许可证发布。

## 开发环境

使用 Windows 10/11 与 64 位 Python 3.12。实时音频功能依赖 Windows 声卡与 WASAPI 回环；离线单元测试尽量使用模拟设备和模拟服务。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\run_defense.bat
```

主程序入口是 `defense_main.py`，主工作台位于 `teams_voice_translator/defense/`。`main.py` 和 `run.bat` 保留经典课堂模式；修改共用音频模块时需要兼顾两个入口。

## 测试

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONPATH = Join-Path $PWD 'tests'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

录音续录相关的定向测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_defense_recording_continuity.py' -v
```

离线测试不能替代真实声卡、会议路由或云端音色试听。请明确记录验证范围；需要真实模型调用的验证会产生费用，应与离线测试分开。

## 构建

```powershell
# 本机使用，输出 dist\DefenseMode\DefenseMode.exe
.\build_defense.ps1

# 分享给他人，排除个人主题与数据
.\build_share.ps1
```

分享版脚本会准备构建依赖、检查个人文件并生成压缩包。不要把自己完整的数据目录或本机调试文件夹打包发布。

## 提交要求

- 描述触发问题的操作、修复后的行为，以及运行过的测试。
- 音频问题注明 Windows 版本、设备名称、会议软件与输入输出路由。
- 保留已渲染音频和已录文件，修改缓存键、参数或录音生命周期时避免让用户重复付费或丢数据。
- UI 修改以清楚的状态和可见反馈为主；名称统一使用「留子答辩器」。为兼容旧数据，包名、EXE 文件名和配置目录不必跟随显示名称迁移。
- 不提交真实 API Key、voice_id、会议内容、论文、讲稿、问答库、样音、字幕、缓存或内存快照。示例和测试使用虚构数据。

新增第三方素材时，一并给出来源和许可证；环境音素材署名文件位于 `teams_voice_translator/ambience_assets/CREDITS.md`。历史更新记录保留在 [CHANGELOG.md](CHANGELOG.md)。
