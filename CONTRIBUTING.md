# 参与贡献

感谢你改进 Teams 双向课堂翻译。提交代码即表示你同意贡献内容按本项目的 GPL-3.0-or-later 许可证发布。

## 本地开发

要求 Windows 10/11、Python 3.12 和可用的音频输入/输出设备。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\run.bat
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

构建 Windows 发行版：

```powershell
.\build.ps1
```

## 提交规范

1. 不要提交 API Key、Workspace ID、录音、字幕、日志或个人会议数据。
2. 一个 Pull Request 尽量只解决一个主题，并说明行为变化和验证方法。
3. 涉及 UI 或音频链路时，请注明 Windows 版本、音频设备和复现步骤。
4. 新功能应补充测试或说明无法自动测试的原因。

## 隐私与合规

录制、转写或处理他人的声音前，请先取得对方同意，并遵守所在地法律、学校或组织规定以及所使用云服务的条款。
