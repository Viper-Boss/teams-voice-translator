答辩实时语音翻译工具 · 使用说明（分享版 v{VERSION}）
==================================================

一、第一次打开
1. 先把整个压缩包解压到一个文件夹（比如 D:\DefenseMode），不要在压缩包里直接双击运行。
2. 双击 DefenseMode\DefenseMode.exe。
3. 如果 Windows 弹出「已保护你的电脑」：点「更多信息」→「仍要运行」。这是因为程序没有数字签名，不是病毒。
   杀毒软件误报时，把解压后的文件夹加入信任即可。

二、需要准备的东西（每个人都要用自己的）
1. 阿里云百炼 API Key：到 https://bailian.console.aliyun.com 开通并创建 API Key，
   在程序「设置 → 账号与网络」里填写 Workspace ID 和 API Key。费用记在你自己的账号上。
   ※ 这个程序里没有任何人的 API Key，也不会上传你的 Key。
2. 自己的克隆音色：「设置 → 声音」里点「一键克隆音色…」，按提示录一段自己的声音，生成后自动填好。
3. 虚拟声卡 VB-CABLE（把翻译后的声音送进 Teams / 腾讯会议）：
   到 https://vb-audio.com/Cable/ 下载安装，装完重启电脑。
   然后在 Teams 里把「麦克风」选成「CABLE Output」，程序设置里的「程序输出」选「CABLE Input」。
4. 耳机：建议戴耳机，避免评委的声音被自己的麦克风再录进去。

三、常用按键（设置 →「快捷键」里可以改）
- F8：原声直通（直接用自己的声音说话）
- F9：按住说中文，松开后翻成英文、用克隆音色读出来
- Esc：停止当前语音（放 PPT 时 Esc 会退出放映，可以在设置里改成别的键）

四、数据存在哪里
设置、问答库、讲稿、渲染好的语音都在本机：%APPDATA%\TeamsVoiceTranslator
（在文件资源管理器地址栏输入 %APPDATA%\TeamsVoiceTranslator 回车就能打开）。
程序本身不带任何别人的数据。

五、出问题时
程序闪退或报错，把 %APPDATA%\TeamsVoiceTranslator\defense_crash.log 发给把程序给你的人。
