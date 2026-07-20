"""
屏幕截取模块 - 支持可调节截取框

功能：
- 基于 mss 库的高性能屏幕截取
- 半透明可拖拽截取框调节界面（基于 tkinter）
- 截取坐标持久化到配置文件
- 多显示器支持
"""

import ctypes
import logging
import tkinter as tk
from tkinter import messagebox
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ScreenCapture:
    """屏幕截取器，支持可拖拽调节的截取框"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化屏幕截取器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()

        # ============ DPI感知设置（必须在任何窗口创建前调用） ============
        # 使 GetSystemMetrics / tkinter / mss 都使用一致的物理像素坐标
        # 避免系统DPI缩放（125%/150%）导致坐标不一致
        self._enable_dpi_awareness()

        # 截取区域参数
        cap_cfg = self.config.get("capture", {})
        region = cap_cfg.get("region", {})
        self.region = {
            "left": region.get("left", 0),
            "top": region.get("top", 800),
            "width": region.get("width", 220),
            "height": region.get("height", 220),
        }
        self.monitor = cap_cfg.get("monitor", 0)
        self.target_fps = cap_cfg.get("fps", 10)

        # 初始化 mss 截图器
        self.sct = mss.mss()
        self.monitors = self.sct.monitors

        # 帧率控制
        self._frame_interval = 1.0 / max(self.target_fps, 1)

        logger.info(f"屏幕截取器初始化完成，截取区域: {self.region}")

    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"配置文件解析错误: {e}")
            return {}

    @staticmethod
    def _enable_dpi_awareness() -> None:
        """
        启用DPI感知，使坐标系统使用物理像素

        必须在任何tkinter窗口创建之前调用。
        避免系统DPI缩放（125%/150%）导致 tkinter 坐标与 mss 截图坐标不一致。
        """
        try:
            # Windows 8.1+: 每监视器DPI感知
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            logger.debug("DPI感知已启用 (SetProcessDpiAwareness=1)")
            return
        except (AttributeError, OSError):
            pass
        try:
            # Windows Vista/7: 系统DPI感知
            ctypes.windll.user32.SetProcessDPIAware()
            logger.debug("DPI感知已启用 (SetProcessDPIAware)")
        except (AttributeError, OSError):
            logger.debug("DPI感知设置失败，使用系统默认")

    def _save_region(self) -> None:
        """保存截取区域坐标到配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError):
            config = {}

        config.setdefault("capture", {})["region"] = self.region

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            logger.info(f"截取区域已保存: {self.region}")
        except IOError as e:
            logger.error(f"保存截取区域失败: {e}")

    def capture(self) -> np.ndarray:
        """
        截取当前屏幕指定区域

        Returns:
            numpy数组 (H, W, 3) BGR格式
        """
        monitor_region = {
            "left": self.region["left"],
            "top": self.region["top"],
            "width": self.region["width"],
            "height": self.region["height"],
        }

        # 使用 mss 截取
        sct_img = self.sct.grab(monitor_region)

        # 转换为 numpy 数组 (BGRA -> BGR)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    def get_region(self) -> dict:
        """
        获取当前截取框坐标

        Returns:
            {"left": int, "top": int, "width": int, "height": int}
        """
        return dict(self.region)

    def get_fps(self) -> float:
        """获取目标帧率"""
        return self.target_fps

    @property
    def frame_interval(self) -> float:
        """获取帧间隔（秒）"""
        return self._frame_interval

    # ==================== 截取框调节界面 ====================

    def open_selector(self) -> None:
        """
        打开可拖拽的截取框调节界面

        操作说明:
        - 鼠标左键拖拽框体中间：移动截取框位置
        - 鼠标左键拖拽边框/四角：调整截取框大小
        - Enter：确认并保存
        - Esc：取消并恢复
        - R：重置为默认位置（屏幕中央偏下）
        """
        selector = _RegionSelector(self)
        selector.run()
        # 更新 region（selector 会直接修改 self.region）


class _RegionSelector:
    """截取框调节器（内部类，基于 tkinter）"""

    # 外观常量
    BORDER_WIDTH = 2
    HANDLE_SIZE = 10
    OVERLAY_COLOR = "#00ff00"
    OVERLAY_ALPHA = 0.3

    # 拖拽阈值
    DRAG_THRESHOLD = 3

    def __init__(self, capture: ScreenCapture):
        self.capture = capture
        self.region = dict(capture.region)  # 工作副本

        # 屏幕尺寸（物理像素—DPI感知已在 ScreenCapture 中设置）
        user32 = ctypes.windll.user32
        self.screen_w = user32.GetSystemMetrics(0)
        self.screen_h = user32.GetSystemMetrics(1)

        # 拖拽状态
        self._drag_mode = None       # "move" / "resize_n" / "resize_s" / ...
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_region = {}
        self._confirmed = False

    def run(self) -> bool:
        """运行调节界面，返回是否确认保存"""
        self._build_window()
        self.root.mainloop()
        return self._confirmed

    def _build_window(self) -> None:
        """构建半透明调节窗口"""
        self.root = tk.Tk()
        self.root.title("截取框调节 - 王者荣耀智能决策系统")
        self.root.attributes("-alpha", 0.7)
        self.root.attributes("-topmost", True)

        # 设置窗口覆盖整个屏幕
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        # ============ 关键修复2: 无边框窗口 ============
        # 去掉标题栏，使 canvas 坐标 = 屏幕物理坐标，消除~30px偏移
        self.root.overrideredirect(True)  # 无边框全屏
        self.root.focus_force()           # 强制获得键盘焦点（接收 Enter/Esc/R）

        # 画布
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定事件（同时绑到canvas和root，确保无边框下键盘事件正常工作）
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Return>", self._on_confirm)
        self.canvas.bind("<Escape>", self._on_cancel)
        self.canvas.bind("<r>", self._on_reset)
        self.canvas.bind("<R>", self._on_reset)
        # root层也绑一份，双重保险
        self.root.bind("<Return>", self._on_confirm)
        self.root.bind("<Escape>", self._on_cancel)
        self.root.bind("<r>", self._on_reset)
        self.root.bind("<R>", self._on_reset)

        # 让canvas获得键盘焦点（无边框窗口必需）
        self.canvas.focus_set()

        # 绘制帮助文字和截取框
        self._draw_hint()
        self._draw_region()

    def _draw_hint(self) -> None:
        """绘制操作提示"""
        hints = [
            "■ 王者荣耀智能决策系统 - 截取框调节",
            "",
            "操作说明:",
            "  • 拖拽绿色边框中间 → 移动截取框",
            "  • 拖拽边框/四角 → 调整大小",
            "  • Enter = 确认保存    Esc = 取消    R = 重置",
            "",
            "请将截取框精确对准游戏小地图区域",
        ]
        hint_text = "\n".join(hints)
        self.canvas.create_text(
            10, 10,
            text=hint_text,
            anchor="nw",
            fill="white",
            font=("Microsoft YaHei", 11),
        )

    def _draw_region(self) -> None:
        """绘制截取框"""
        # 清除旧框
        self.canvas.delete("region")

        x1 = self.region["left"]
        y1 = self.region["top"]
        x2 = x1 + self.region["width"]
        y2 = y1 + self.region["height"]

        # 半透明填充
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill="", outline=self.OVERLAY_COLOR,
            width=self.BORDER_WIDTH,
            tags="region",
        )

        # 四角拖拽手柄
        for cx, cy, cursor_name in [
            (x1, y1, "size_nw_se"), (x2, y1, "size_ne_sw"),
            (x1, y2, "size_ne_sw"), (x2, y2, "size_nw_se"),
        ]:
            self.canvas.create_rectangle(
                cx - self.HANDLE_SIZE, cy - self.HANDLE_SIZE,
                cx + self.HANDLE_SIZE, cy + self.HANDLE_SIZE,
                fill=self.OVERLAY_COLOR, outline="",
                tags="region",
            )

        # 四边中点拖拽手柄
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        for cx, cy, cursor_name in [
            (mid_x, y1, "size_ns"), (mid_x, y2, "size_ns"),
            (x1, mid_y, "size_we"), (x2, mid_y, "size_we"),
        ]:
            self.canvas.create_rectangle(
                cx - 6, cy - 6, cx + 6, cy + 6,
                fill=self.OVERLAY_COLOR, outline="",
                tags="region",
            )

        # 坐标信息
        info = f"({x1}, {y1}) {self.region['width']}×{self.region['height']}"
        self.canvas.create_text(
            mid_x, y1 - 20,
            text=info,
            fill="#00ff00",
            font=("Consolas", 10, "bold"),
            tags="region",
        )

    def _hit_test(self, x: int, y: int) -> Optional[str]:
        """确定鼠标点击位置属于哪个区域"""
        x1 = self.region["left"]
        y1 = self.region["top"]
        x2 = x1 + self.region["width"]
        y2 = y1 + self.region["height"]
        margin = self.HANDLE_SIZE + 4

        # 四角
        if abs(x - x1) < margin and abs(y - y1) < margin:
            return "resize_nw"
        if abs(x - x2) < margin and abs(y - y1) < margin:
            return "resize_ne"
        if abs(x - x1) < margin and abs(y - y2) < margin:
            return "resize_sw"
        if abs(x - x2) < margin and abs(y - y2) < margin:
            return "resize_se"

        # 四边
        if x1 <= x <= x2 and abs(y - y1) < margin:
            return "resize_n"
        if x1 <= x <= x2 and abs(y - y2) < margin:
            return "resize_s"
        if y1 <= y <= y2 and abs(x - x1) < margin:
            return "resize_w"
        if y1 <= y <= y2 and abs(x - x2) < margin:
            return "resize_e"

        # 内部
        if x1 < x < x2 and y1 < y < y2:
            return "move"

        return None

    def _on_press(self, event: tk.Event) -> None:
        """鼠标按下"""
        self._drag_mode = self._hit_test(event.x, event.y)
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_start_region = dict(self.region)

    def _on_drag(self, event: tk.Event) -> None:
        """鼠标拖拽"""
        if not self._drag_mode:
            return

        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        r = self._drag_start_region

        if self._drag_mode == "move":
            new_left = r["left"] + dx
            new_top = r["top"] + dy
            # 边界限制
            new_left = max(0, min(new_left, self.screen_w - r["width"]))
            new_top = max(0, min(new_top, self.screen_h - r["height"]))
            self.region["left"] = new_left
            self.region["top"] = new_top

        elif self._drag_mode == "resize_nw":
            self.region["left"] = max(0, min(r["left"] + dx, r["left"] + r["width"] - 40))
            self.region["top"] = max(0, min(r["top"] + dy, r["top"] + r["height"] - 40))
            self.region["width"] = r["left"] + r["width"] - self.region["left"]
            self.region["height"] = r["top"] + r["height"] - self.region["top"]

        elif self._drag_mode == "resize_ne":
            self.region["top"] = max(0, min(r["top"] + dy, r["top"] + r["height"] - 40))
            self.region["width"] = max(40, min(r["width"] + dx, self.screen_w - r["left"]))
            self.region["height"] = r["top"] + r["height"] - self.region["top"]

        elif self._drag_mode == "resize_sw":
            self.region["left"] = max(0, min(r["left"] + dx, r["left"] + r["width"] - 40))
            self.region["width"] = r["left"] + r["width"] - self.region["left"]
            self.region["height"] = max(40, min(r["height"] + dy, self.screen_h - r["top"]))

        elif self._drag_mode == "resize_se":
            self.region["width"] = max(40, min(r["width"] + dx, self.screen_w - r["left"]))
            self.region["height"] = max(40, min(r["height"] + dy, self.screen_h - r["top"]))

        elif self._drag_mode == "resize_n":
            self.region["top"] = max(0, min(r["top"] + dy, r["top"] + r["height"] - 40))
            self.region["height"] = r["top"] + r["height"] - self.region["top"]

        elif self._drag_mode == "resize_s":
            self.region["height"] = max(40, min(r["height"] + dy, self.screen_h - r["top"]))

        elif self._drag_mode == "resize_w":
            self.region["left"] = max(0, min(r["left"] + dx, r["left"] + r["width"] - 40))
            self.region["width"] = r["left"] + r["width"] - self.region["left"]

        elif self._drag_mode == "resize_e":
            self.region["width"] = max(40, min(r["width"] + dx, self.screen_w - r["left"]))

        self._draw_region()

    def _on_release(self, event: tk.Event) -> None:
        """鼠标释放"""
        self._drag_mode = None

    def _on_confirm(self, event: tk.Event) -> None:
        """确认并保存"""
        self._confirmed = True
        # 更新 capture 的 region
        self.capture.region = dict(self.region)
        self.capture._save_region()
        logger.info("截取框已确认保存")
        self.root.destroy()

    def _on_cancel(self, event: tk.Event) -> None:
        """取消"""
        self._confirmed = False
        logger.info("截取框调节已取消")
        self.root.destroy()

    def _on_reset(self, event: tk.Event) -> None:
        """重置为默认位置（屏幕中央偏下，适合小地图位置）"""
        # 默认 220x220 放在屏幕右下区域
        default_w = 220
        default_h = 220
        self.region = {
            "left": self.screen_w - default_w - 50,
            "top": self.screen_h - default_h - 100,
            "width": default_w,
            "height": default_h,
        }
        self._drag_start_region = dict(self.region)
        self._draw_region()
        logger.info("截取框已重置为默认位置")


# ==================== 便捷截图（不使用 mss，用于快速测试） ====================

def quick_capture(region: dict) -> np.ndarray:
    """
    使用 mss 快速截取指定区域（独立函数，不依赖配置文件）

    Args:
        region: {"left": int, "top": int, "width": int, "height": int}

    Returns:
        numpy数组 (H, W, 3) BGR格式
    """
    with mss.mss() as sct:
        sct_img = sct.grab(region)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
