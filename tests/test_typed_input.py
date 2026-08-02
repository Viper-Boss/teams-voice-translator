import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from teams_voice_translator.ui import SendTextEdit


class TypedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_enter_sends_without_inserting_newline(self):
        edit = SendTextEdit()
        edit.setPlainText("你好")
        calls = []
        edit.send_requested.connect(lambda: calls.append(True))
        QTest.keyClick(edit, Qt.Key_Return)
        self.assertEqual(calls, [True])
        self.assertEqual(edit.toPlainText(), "你好")

    def test_ctrl_enter_inserts_newline_without_sending(self):
        edit = SendTextEdit()
        edit.setPlainText("第一行")
        edit.moveCursor(edit.textCursor().MoveOperation.End)
        calls = []
        edit.send_requested.connect(lambda: calls.append(True))
        QTest.keyClick(edit, Qt.Key_Return, Qt.ControlModifier)
        self.assertEqual(calls, [])
        self.assertEqual(edit.toPlainText(), "第一行\n")


if __name__ == "__main__":
    unittest.main()
