"""
Real-time training chart widget using pyqtgraph for high-performance plotting.
"""
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from collections import deque
import numpy as np


class TrainingChartWidget(QWidget):
    """Displays loss/accuracy curves that update in real time during training."""

    def __init__(self, title: str = "训练曲线", max_points: int = 5000, parent=None):
        super().__init__(parent)
        self._max_points = max_points
        self._step_data = deque(maxlen=max_points)
        self._loss_data = deque(maxlen=max_points)
        self._asr_loss_data = deque(maxlen=max_points)
        self._dialect_loss_data = deque(maxlen=max_points)
        self._speaker_loss_data = deque(maxlen=max_points)
        self._accuracy_data = deque(maxlen=max_points)
        self._lr_data = deque(maxlen=max_points)
        self._step_counter = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create graphics layout widget
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#ffffff")

        # === Top plot: Loss ===
        self._loss_plot = self._glw.addPlot(row=0, col=0)
        self._loss_plot.setTitle("Loss", color="#2c3e50", size="12pt")
        self._loss_plot.setLabel("left", "Loss")
        self._loss_plot.setLabel("bottom", "Step")
        self._loss_plot.showGrid(x=True, y=True, alpha=0.3)
        # 横向单行图例 (4 条损失曲线并排)，避免纵向堆叠在小图中被裁掉
        self._loss_plot.addLegend(offset=(-10, 10), colCount=4, labelTextSize='8pt')

        self._total_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(color="#1a73e8", width=1.5), name="Total Loss"
        )
        self._asr_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(color="#28a745", width=1), name="ASR Loss"
        )
        self._dialect_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(color="#e6a817", width=1), name="Dialect Loss"
        )
        self._speaker_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(color="#dc3545", width=1), name="Speaker Loss"
        )

        # === Bottom plot: Accuracy + LR ===
        self._acc_plot = self._glw.addPlot(row=1, col=0)
        self._acc_plot.setTitle("Accuracy / Learning Rate", color="#2c3e50", size="12pt")
        self._acc_plot.setLabel("left", "Accuracy")
        self._acc_plot.setLabel("bottom", "Step")
        self._acc_plot.showGrid(x=True, y=True, alpha=0.3)

        self._acc_curve = self._acc_plot.plot(
            pen=pg.mkPen(color="#1a73e8", width=1.5), name="Accuracy"
        )

        # LR on right axis
        self._lr_axis = pg.ViewBox()
        self._lr_axis.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self._acc_plot.scene().addItem(self._lr_axis)
        self._acc_plot.getAxis('right').linkToView(self._lr_axis)
        self._acc_plot.getAxis('right').setLabel("LR", color="#9aa0a6")
        self._lr_curve = pg.PlotCurveItem(pen=pg.mkPen(color="#9aa0a6", width=1, style=Qt.DashLine))
        self._lr_axis.addItem(self._lr_curve)
        self._acc_plot.vb.sigResized.connect(self._update_lr_view)

        layout.addWidget(self._glw)

    def _update_lr_view(self):
        """Sync LR axis geometry with accuracy plot."""
        self._lr_axis.setGeometry(self._acc_plot.vb.sceneBoundingRect())
        self._lr_axis.linkedViewChanged(self._acc_plot.vb, self._lr_axis.XAxis)

    def add_step(self, step: int, loss: float, accuracy: float = 0,
                 asr_loss: float = 0, dialect_loss: float = 0,
                 speaker_loss: float = 0, lr: float = 0):
        """Add a training step data point. Called from TrainingWorker signals.

        Args:
            step: Current training step
            loss: Total loss
            accuracy: Current accuracy (0 if not evaluated yet)
            asr_loss, dialect_loss, speaker_loss: Per-task losses
            lr: Current learning rate
        """
        self._step_counter = step
        self._step_data.append(step)
        self._loss_data.append(loss)
        self._asr_loss_data.append(asr_loss)
        self._dialect_loss_data.append(dialect_loss)
        self._speaker_loss_data.append(speaker_loss)
        if accuracy > 0:
            self._accuracy_data.append((step, accuracy))
        if lr > 0:
            self._lr_data.append((step, lr))

        # Update curves
        steps_arr = np.array(self._step_data, dtype=int)

        self._total_loss_curve.setData(steps_arr, np.array(self._loss_data, dtype=float))
        self._asr_loss_curve.setData(steps_arr, np.array(self._asr_loss_data, dtype=float))
        self._dialect_loss_curve.setData(steps_arr, np.array(self._dialect_loss_data, dtype=float))
        self._speaker_loss_curve.setData(steps_arr, np.array(self._speaker_loss_data, dtype=float))

        if self._accuracy_data:
            acc_steps, acc_vals = zip(*self._accuracy_data)
            self._acc_curve.setData(np.array(acc_steps), np.array(acc_vals))

        if self._lr_data:
            lr_steps, lr_vals = zip(*self._lr_data)
            self._lr_curve.setData(np.array(lr_steps), np.array(lr_vals))

    def add_epoch_marker(self, epoch: int):
        """Add a vertical marker at epoch boundary."""
        color = pg.mkColor("#e0e4e8")
        self._loss_plot.addItem(
            pg.InfiniteLine(pos=self._step_counter, angle=90, pen=pg.mkPen(color=color, width=1, style=Qt.DashLine))
        )

    def clear(self):
        """Reset all chart data."""
        self._step_data.clear()
        self._loss_data.clear()
        self._asr_loss_data.clear()
        self._dialect_loss_data.clear()
        self._speaker_loss_data.clear()
        self._accuracy_data.clear()
        self._lr_data.clear()
        self._step_counter = 0
        self._total_loss_curve.clear()
        self._asr_loss_curve.clear()
        self._dialect_loss_curve.clear()
        self._speaker_loss_curve.clear()
        self._acc_curve.clear()
        self._lr_curve.clear()
