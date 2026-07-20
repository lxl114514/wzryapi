"""
决策结果显示模块 - 悬浮窗显示战术决策

功能：
- 无边框半透明悬浮窗，始终置顶（基于 tkinter）
- 按优先级配色显示决策
- 自动淡出效果
- 历史记录查看
- 位置可配置

线程安全设计：
- tkinter 窗口运行在独立线程中
- 主线程通过 queue.Queue 将决策数据传递给 tkinter 线程
- 所有 tkinter 操作始终在 tkinter 线程内执行
"""

import logging
import queue
import threading
import time
import tkinter as tk
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==================== 颜色与样式常量 ====================

PRIORITY_COLORS = {
    "high": "#ff4444",
    "medium": "#ffaa00",
    "low": "#44dd44",
    "unknown": "#aaaaaa",
}

PRIORITY_LABELS = {
    "high": "⚠ 高优先级",
    "medium": "● 中优先级",
    "low": "○ 低优先级",
    "unknown": "- 未知",
}

ACTION_ICONS = {
    "进攻": "⚔️",
    "撤退": "🏃",
    "发育": "💰",
    "打龙": "🐉",
    "推塔": "🗼",
    "支援": "🤝",
    "蹲守": "👀",
    "反野": "🐗",
    "清线": "🧹",
    "回城": "🏠",
    "观望": "⏳",
}


class DecisionDisplay:
    """悬浮窗决策显示器（线程安全）"""

    def __init__(
        self,
        position: str = "top-right",
        opacity: float = 0.85,
        font_size: int = 16,
        show_reason: bool = True,
        highlight_duration: float = 3.0,
        colors: Optional[Dict[str, str]] = None,
        width: int = 300,
        custom_x: int = 100,
        custom_y: int = 100,
    ):
        """
        初始化决策显示器

        Args:
            position: 窗口位置 (top-left / top-right / bottom-left / bottom-right / custom)
            opacity: 窗口透明度 0.0~1.0
            font_size: 字体大小
            show_reason: 是否显示理由
            highlight_duration: 高亮持续时间（秒）
            colors: 优先级颜色覆盖
            width: 窗口宽度（像素）
            custom_x: 自定义X坐标
            custom_y: 自定义Y坐标
        """
        self.position = position
        self.opacity = opacity
        self.font_size = font_size
        self.show_reason = show_reason
        self.highlight_duration = highlight_duration
        self.width = width
        self.custom_x = custom_x
        self.custom_y = custom_y

        # 颜色
        self.colors = dict(PRIORITY_COLORS)
        if colors:
            self.colors.update(colors)

        # 历史记录（最近5条）
        self.history: deque = deque(maxlen=5)

        # === 线程安全通信 ===
        # 主线程 → tkinter线程的决策队列
        self._decision_queue: queue.Queue = queue.Queue(maxsize=32)

        # tkinter 状态
        self._root: Optional[tk.Tk] = None
        self._running = False
        self._visible = True
        self._highlight_alpha = 1.0
        self._highlight_start = 0.0
        self._current_decision: Optional[Dict] = None

        # 线程
        self._thread: Optional[threading.Thread] = None

        logger.info(f"决策显示器初始化: position={position}, opacity={opacity}")

    @classmethod
    def from_config(cls, config_path: str) -> "DecisionDisplay":
        """从配置文件创建显示器实例"""
        import yaml

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"加载配置文件失败: {e}")
            config = {}

        display_cfg = config.get("display", {})
        return cls(
            position=display_cfg.get("position", "top-right"),
            opacity=display_cfg.get("opacity", 0.85),
            font_size=display_cfg.get("font_size", 18),
            show_reason=display_cfg.get("show_reason", True),
            highlight_duration=display_cfg.get("highlight_duration", 3),
            colors=display_cfg.get("colors"),
            width=display_cfg.get("width", 300),
            custom_x=display_cfg.get("custom_x", 100),
            custom_y=display_cfg.get("custom_y", 100),
        )

    # ==================== 启动与关闭 ====================

    def start(self) -> None:
        """启动悬浮窗（在独立线程中运行tkinter主循环）"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._tk_mainloop, daemon=True)
        self._thread.start()
        # 等待 tkinter 窗口创建完毕
        time.sleep(0.5)
        logger.info("决策显示悬浮窗已启动")

    def stop(self) -> None:
        """停止悬浮窗"""
        self._running = False
        if self._root:
            try:
                self._root.quit()  # 退出 mainloop（线程安全）
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("决策显示悬浮窗已停止")

    def _tk_mainloop(self) -> None:
        """tkinter 主循环（运行在独立线程中）"""
        self._root = tk.Tk()
        self._root.title("战术决策")
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", self.opacity)
        self._root.overrideredirect(True)

        # 构建UI
        self._build_ui()

        # 设置窗口位置
        self._position_window()

        # 启动定时任务：轮询决策队列 + 淡出效果
        self._poll_queue()
        self._update_fade()

        # 进入 tkinter 事件循环（阻塞，直到窗口关闭）
        try:
            self._root.mainloop()
        except Exception:
            pass

        try:
            self._root.destroy()
        except Exception:
            pass

    def _poll_queue(self) -> None:
        """从队列中取决策并更新UI（在tkinter线程中运行）"""
        try:
            while True:
                decision = self._decision_queue.get_nowait()
                self._update_decision_ui(decision)
        except queue.Empty:
            pass
        # 每50ms检查一次队列
        if self._running and self._root:
            try:
                self._root.after(50, self._poll_queue)
            except tk.TclError:
                pass

    def _build_ui(self) -> None:
        """构建悬浮窗UI组件"""
        self._main_frame = tk.Frame(
            self._root,
            bg="#1a1a2e",
            highlightthickness=1,
            highlightbackground="#383366",
        )
        self._main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题栏（可拖拽）
        self._title_bar = tk.Frame(self._main_frame, bg="#16213e", height=24)
        self._title_bar.pack(fill=tk.X)
        self._title_bar.pack_propagate(False)
        self._title_bar.bind("<ButtonPress-1>", self._on_title_press)
        self._title_bar.bind("<B1-Motion>", self._on_title_drag)

        self._title_label = tk.Label(
            self._title_bar,
            text="🎮 战术决策",
            fg="#e0e0e0",
            bg="#16213e",
            font=("Microsoft YaHei", 9, "bold"),
        )
        self._title_label.pack(side=tk.LEFT, padx=6, pady=2)

        # 关闭按钮
        close_btn = tk.Label(
            self._title_bar,
            text="✕",
            fg="#888888",
            bg="#16213e",
            font=("Arial", 10),
            cursor="hand2",
        )
        close_btn.pack(side=tk.RIGHT, padx=6, pady=2)
        close_btn.bind("<Button-1>", lambda e: self.hide())

        # 内容区域
        self._content_frame = tk.Frame(self._main_frame, bg="#1a1a2e")
        self._content_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # 决策显示标签（suggestion 一句话决策）
        self._action_label = tk.Label(
            self._content_frame,
            text="等待战局分析...",
            fg="#a0a0a0",
            bg="#1a1a2e",
            font=("Microsoft YaHei", self.font_size),
            wraplength=self.width - 20,
            justify=tk.LEFT,
        )
        self._action_label.pack(anchor="w")

        self._target_label = tk.Label(
            self._content_frame,
            text="",
            fg="#c0c0c0",
            bg="#1a1a2e",
            font=("Microsoft YaHei", self.font_size - 2),
            wraplength=self.width - 20,
            justify=tk.LEFT,
        )
        self._target_label.pack(anchor="w")

        # 理由标签
        self._reason_label = tk.Label(
            self._content_frame,
            text="",
            fg="#888888",
            bg="#1a1a2e",
            font=("Microsoft YaHei", 9),
            wraplength=self.width - 20,
            justify=tk.LEFT,
        )
        if self.show_reason:
            self._reason_label.pack(anchor="w", pady=(2, 0))

        # 分隔线
        self._separator = tk.Frame(self._main_frame, bg="#333366", height=1)
        self._separator.pack(fill=tk.X, padx=10)

        # 历史记录区域（可折叠）
        self._history_frame = tk.Frame(self._main_frame, bg="#1a1a2e")
        self._history_visible = False
        self._toggle_btn = tk.Label(
            self._main_frame,
            text="▼ 历史记录",
            fg="#6666aa",
            bg="#1a1a2e",
            font=("Microsoft YaHei", 8),
            cursor="hand2",
        )
        self._toggle_btn.pack(fill=tk.X, padx=10, pady=(2, 0))
        self._toggle_btn.bind("<Button-1>", self._toggle_history)

    def _position_window(self) -> None:
        """根据配置映射窗口位置"""
        self._root.update_idletasks()
        win_w = self.width
        win_h = 150

        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()

        positions = {
            "top-left": (20, 40),
            "top-right": (screen_w - win_w - 20, 40),
            "bottom-left": (20, screen_h - win_h - 60),
            "bottom-right": (screen_w - win_w - 20, screen_h - win_h - 60),
            "custom": (self.custom_x, self.custom_y),
        }

        x, y = positions.get(self.position, positions["top-right"])
        self._root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    # ==================== 线程安全的决策推送接口 ====================

    def show_decision(self, decision: Dict[str, Any]) -> None:
        """
        显示一条决策（线程安全——通过队列传递给tkinter线程）

        策略：如果新决策是"观望"但之前已有有效决策，则保持上次决策不变
        Args:
            decision: 决策字典，包含 action, target, reason, priority, suggestion
        """
        if not self._running:
            self.start()

        action = decision.get("action", "观望")

        # === 关键：不覆盖上一次的有效决策 ===
        # 如果新决策是"观望"且之前已有非观望的有效决策，则跳过更新
        if action in ("观望", "等待") and self._current_decision:
            prev_action = self._current_decision.get("action", "")
            if prev_action not in ("观望", "等待", ""):
                logger.debug(f"跳过'观望'决策，保持上一次: {prev_action}")
                return

        self._current_decision = decision
        self.history.appendleft(dict(decision))

        # 将决策放入队列，由 tkinter 线程异步取出并更新 UI
        try:
            self._decision_queue.put_nowait(decision)
        except queue.Full:
            # 队列满时丢弃旧决策，腾出空间给新决策
            try:
                self._decision_queue.get_nowait()
                self._decision_queue.put_nowait(decision)
            except queue.Empty:
                pass

        logger.info(f"显示决策: {action} → {decision.get('target', '')}")

    def update_decision(self, decision: Dict[str, Any]) -> None:
        """更新当前决策（别名）"""
        self.show_decision(decision)

    def _update_decision_ui(self, decision: Dict[str, Any]) -> None:
        """在tkinter线程中更新UI（由 _poll_queue 调用）"""
        try:
            priority = decision.get("priority", "low")
            action = decision.get("action", "观望")
            suggestion = decision.get("suggestion", "")
            color = self.colors.get(priority, self.colors["low"])
            icon = ACTION_ICONS.get(action, "📋")

            # 优先显示完整的一句话决策（suggestion），否则回退到action+target格式
            if suggestion:
                display_text = f"{icon} {suggestion}"
                self._action_label.config(
                    text=display_text,
                    fg=color,
                    wraplength=self.width - 20,
                )
                self._target_label.config(text="")
                if self.show_reason:
                    self._reason_label.config(text="")
                logger.debug(f"UI更新: {display_text}")
            else:
                target = decision.get("target", "")
                reason = decision.get("reason", "")
                display_text = f"{icon} {action} → {target}"
                self._action_label.config(
                    text=f"{icon} {action}",
                    fg=color,
                )
                self._target_label.config(text=f"→ {target}" if target else "", fg="#c0c0c0")
                if self.show_reason:
                    self._reason_label.config(text=reason[:50] if reason else "", fg="#888888")
                logger.debug(f"UI更新(旧格式): {display_text}")

            priority_text = PRIORITY_LABELS.get(priority, "")
            self._title_label.config(text=f"🎮 {priority_text}", fg=color)

            self._highlight_alpha = 1.0
            self._highlight_start = time.time()
            self._root.attributes("-alpha", self.opacity)

            # 更新完内容后让窗口自适应高度
            self._root.update_idletasks()
            self._root.geometry("")

            if self._history_visible:
                self._update_history_display()
        except Exception as e:
            logger.warning(f"更新决策UI失败: {e}")

    # ==================== 显示控制 ====================

    def show(self) -> None:
        """显示悬浮窗"""
        self._visible = True
        if self._root:
            try:
                self._root.deiconify()
            except Exception:
                pass

    def hide(self) -> None:
        """隐藏悬浮窗"""
        self._visible = False
        if self._root:
            try:
                self._root.withdraw()
            except Exception:
                pass

    def toggle(self) -> None:
        """切换显示/隐藏"""
        if self._visible:
            self.hide()
        else:
            self.show()

    def close(self) -> None:
        """关闭显示器"""
        self.stop()

    # ==================== 淡出效果 ====================

    def _update_fade(self) -> None:
        """定时更新淡出效果"""
        if not self._running or not self._root:
            return

        try:
            if self._highlight_alpha < self.opacity:
                elapsed = time.time() - self._highlight_start
                if elapsed >= self.highlight_duration:
                    fade_duration = 1.5
                    if elapsed < self.highlight_duration + fade_duration:
                        progress = (elapsed - self.highlight_duration) / fade_duration
                        self._highlight_alpha = max(
                            self.opacity - 0.3, 1.0 - progress * 0.3,
                        )
                    else:
                        self._highlight_alpha = self.opacity

                if self._highlight_alpha < self.opacity:
                    self._root.attributes("-alpha", self._highlight_alpha)

            self._root.after(100, self._update_fade)
        except tk.TclError:
            pass

    # ==================== 历史记录 ====================

    def _toggle_history(self, event=None) -> None:
        """切换历史记录显示"""
        self._history_visible = not self._history_visible
        if self._history_visible:
            self._toggle_btn.config(text="▲ 历史记录")
            self._history_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
            self._update_history_display()
        else:
            self._toggle_btn.config(text="▼ 历史记录")
            self._history_frame.pack_forget()
        self._root.update_idletasks()
        self._root.geometry("")

    def _update_history_display(self) -> None:
        """更新历史记录显示"""
        for widget in self._history_frame.winfo_children():
            widget.destroy()

        if not self.history:
            tk.Label(
                self._history_frame,
                text="暂无历史记录",
                fg="#555555",
                bg="#1a1a2e",
                font=("Microsoft YaHei", 8),
            ).pack(anchor="w")
            return

        for i, decision in enumerate(self.history):
            priority = decision.get("priority", "low")
            suggestion = decision.get("suggestion", "")
            action = decision.get("action", "")
            target = decision.get("target", "")
            color = self.colors.get(priority, self.colors["low"])
            icon = ACTION_ICONS.get(action, "📋")

            # 优先显示 suggestion，否则回退到 action→target
            if suggestion:
                text = f"{icon} {suggestion}"
            else:
                text = f"{icon} {action} → {target}"
            if len(text) > 35:
                text = text[:33] + "..."

            label = tk.Label(
                self._history_frame,
                text=text,
                fg=color,
                bg="#1a1a2e",
                font=("Microsoft YaHei", 8),
                anchor="w",
            )
            label.pack(fill=tk.X, pady=1)
            if i == 0:
                label.config(font=("Microsoft YaHei", 9, "bold"))

    # ==================== 窗口拖拽 ====================

    def _on_title_press(self, event: tk.Event) -> None:
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_title_drag(self, event: tk.Event) -> None:
        if hasattr(self, "_drag_x") and self._root:
            dx = event.x - self._drag_x
            dy = event.y - self._drag_y
            x = self._root.winfo_x() + dx
            y = self._root.winfo_y() + dy
            self._root.geometry(f"+{x}+{y}")
