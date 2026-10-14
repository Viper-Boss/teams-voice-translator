"""Read-only comparison; only checked, fixable answers can be applied."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QTableWidget,QTableWidgetItem,
    QPlainTextEdit,QLabel,QPushButton,QHeaderView,QSplitter)


class AnswerReviewDialog(QDialog):
    def __init__(self,parent,entries,reviews):
        super().__init__(parent)
        self.reviews=reviews; self.entries={e.id:e for e in entries}
        self.setWindowTitle('Plus 检查结果 · 核对后应用');self.resize(1180,800)
        layout=QVBoxLayout(self)
        layout.addWidget(QLabel('勾选要应用的修正。需核实的答案只提示问题，不自动改写；模型检查不能代替论文事实核对。'))
        self.table=QTableWidget(len(reviews),4)
        self.table.setHorizontalHeaderLabels(['应用','编号','结果','问题 / 依据'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        labels={'pass':'未发现问题','fix':'可修正','check':'需核实'}
        for row,r in enumerate(reviews):
            check=QTableWidgetItem('');check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if r['status']=='fix':
                check.setFlags(check.flags() | Qt.ItemIsUserCheckable);check.setCheckState(Qt.Checked)
            self.table.setItem(row,0,check)
            for col,value in enumerate([r['id'],labels[r['status']],'；'.join(r['issues'])+'  '+ '、'.join(r['evidence'])],1):
                self.table.setItem(row,col,QTableWidgetItem(value))
        layout.addWidget(self.table,1)
        panes=QSplitter(Qt.Horizontal)
        self.before=QPlainTextEdit();self.after=QPlainTextEdit()
        self.before.setReadOnly(True);self.after.setReadOnly(True)
        panes.addWidget(self.before);panes.addWidget(self.after);layout.addWidget(panes,1)
        buttons=QHBoxLayout()
        apply=QPushButton('应用勾选的修正');apply.clicked.connect(self.accept)
        cancel=QPushButton('关闭，不修改');cancel.clicked.connect(self.reject)
        buttons.addWidget(apply);buttons.addWidget(cancel);layout.addLayout(buttons)
        self.table.currentCellChanged.connect(self._show)
        if reviews:self.table.setCurrentCell(0,1)

    def _show(self,row,*_):
        if not 0<=row<len(self.reviews):return
        r=self.reviews[row];e=self.entries[r['id']]
        self.before.setPlainText(f'原回答\n\n{e.question}\n\n{e.answer}\n\n{e.answer_zh}')
        self.after.setPlainText('检查：'+'；'.join(r['issues'])+'\n依据：'+'、'.join(r['evidence'])+
            '\n\n修正英文：\n'+r.get('answer','（不修改，请核实）')+'\n\n修正中文：\n'+r.get('answer_zh',''))

    def chosen(self):
        return [r for row,r in enumerate(self.reviews) if r['status']=='fix' and self.table.item(row,0).checkState()==Qt.Checked]
