import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QScrollArea
from teams_voice_translator.defense.ui import DefenseWindow, SettingsDialog, VoiceCompareDialog, CloneVoiceDialog, RehearsalDialog
from teams_voice_translator.defense.settings import DefenseSettings
app=QApplication([])
app.setStyle('Fusion')
from PySide6.QtGui import QFontDatabase
for name in ('msyh.ttc','segoeui.ttf','seguiemj.ttf'):
 QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)

out=Path('test-artifacts/improved-ui-compact' if '--compact' in sys.argv else 'test-artifacts/improved-ui');out.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory() as tmp, patch('teams_voice_translator.defense.ui._safe_list_audio_devices',return_value=([],[])), patch('teams_voice_translator.defense.ui._safe_list_loopback_devices',return_value=[]):
 settings=DefenseSettings(Path(tmp))
 settings.update({'tts_model':'cosyvoice-v3.5-plus','tts_voice_id':'demo-voice','qa_auto':False})
 window=DefenseWindow(settings)
 if '--compact' in sys.argv:window.resize(1060,680)
 window.show();app.processEvents()
 window.grab().save(str(out/'main-empty.png'))
 window.engine._register_segment('me','我的研究将物理约束和漏磁信号相结合，用于缺陷反演。')
 window._on_my_done(1,'My research combines physical constraints with magnetic flux leakage signals for defect inversion.',0)
 window._on_committee_done(2,'How does your method perform under noisy conditions?','你们的方法在有噪声的条件下表现如何？')
 window.live_english.setText('The physical constraints help keep the prediction consistent, even when the measurements contain noise.')
 window.mic_preview.setText('物理约束能够帮助保持预测的一致性，即使测量中存在噪声。')
 app.processEvents();window.grab().save(str(out/'main-demo.png'))
 window.focus_check.setChecked(True);app.processEvents();window.grab().save(str(out/'focus-demo.png'))
 dialog=SettingsDialog(window,settings);dialog.show();app.processEvents();dialog.grab().save(str(out/'settings.png'))
 settings_scroll=dialog.findChild(QScrollArea);settings_scroll.verticalScrollBar().setValue(settings_scroll.verticalScrollBar().maximum());app.processEvents();dialog.grab().save(str(out/'settings-lower.png'));dialog.close()
 compare=VoiceCompareDialog(window,settings);compare.show();app.processEvents()
 compare._show_progress(compare._cancel,compare.text_edit.toPlainText(),.45)
 app.processEvents();compare.grab().save(str(out/'voice-compare.png'))
 compare.subtitle_overlay.grab().save(str(out/'voice-compare-subtitles.png'));compare.close()
 rehearsal=RehearsalDialog(window,settings)
 rehearsal.resize(1100,700)
 rehearsal._set_rows(['我的研究将物理约束与测量信号相结合，以提高有噪声条件下的缺陷反演稳定性。','下面我将介绍主要实验结果，以及这个方法在实际应用中的局限。'],
 ['My research combines physical constraints with the information in the measured signals to improve the stability of defect inversion under noisy conditions.', 'Next, I will present the main experimental results and discuss the limitations of this method in practical applications.'])
 rehearsal.show();app.processEvents();rehearsal._show_play_progress(0,.47);app.processEvents()
 rehearsal.grab().save(str(out/'rehearsal.png'))
 rehearsal.subtitle_overlay.grab().save(str(out/'rehearsal-subtitles.png'));rehearsal.close()
 with patch.object(CloneVoiceDialog, '_maybe_auto_probe_oss'):
  clone=CloneVoiceDialog(window,settings,'cosyvoice-v3.5-plus')
  clone._set_sample(Path('studio-demo.wav'),35.25)
  clone.show();app.processEvents()
  clone.findChild(QScrollArea).ensureWidgetVisible(clone.sample_options)
  app.processEvents();clone.grab().save(str(out/'clone-options.png'));clone.close()
 window.close()
