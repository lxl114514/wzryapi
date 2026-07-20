"""
王者荣耀智能决策系统 - 主程序入口

功能：
- 模块初始化与编排
- 主循环（截图→检测→决策→显示）
- 全局快捷键监听
- CLI参数管理
- 调试模式
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime

import cv2

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screen_capture import ScreenCapture
from src.yolo_detector import YOLODetector
from src.api_decision import APIDecisionMaker
from src.display import DecisionDisplay

# ==================== 日志配置 ====================

def setup_logging(level: str = "INFO") -> None:
    """配置日志系统"""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 降低第三方库日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ==================== 快捷键监听 ====================

class HotkeyListener:
    """全局快捷键监听器（基于pynput）"""

    def __init__(self, callbacks: dict):
        """
        初始化快捷键监听器

        Args:
            callbacks: {"<组合键>": callback_function, ...}
        """
        self.callbacks = callbacks
        self._running = False
        self._listener = None
        self._pressed_keys = set()

    def start(self):
        """启动监听"""
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    if hasattr(key, "char") and key.char:
                        self._pressed_keys.add(key.char)
                    else:
                        self._pressed_keys.add(str(key))
                except Exception:
                    pass

                # 检查快捷键组合
                self._check_hotkeys()

            def on_release(key):
                try:
                    if hasattr(key, "char") and key.char:
                        self._pressed_keys.discard(key.char)
                    else:
                        self._pressed_keys.discard(str(key))
                except Exception:
                    pass

            self._running = True
            self._listener = keyboard.Listener(
                on_press=on_press, on_release=on_release
            )
            self._listener.daemon = True
            self._listener.start()
            logger.info("全局快捷键监听已启动")

        except ImportError:
            logger.warning(
                "未安装 pynput 库，快捷键功能不可用。"
                "安装命令: pip install pynput"
            )

    def _check_hotkeys(self):
        """检查是否触发快捷键"""
        # Ctrl+Shift+S → 重新打开截取框
        if self._is_pressed(["Key.ctrl", "Key.shift", "s"]):
            logger.info("快捷键: 重新打开截取框")
            self.callbacks.get("reopen_selector", lambda: None)()

        # Ctrl+Shift+H → 显示/隐藏悬浮窗
        if self._is_pressed(["Key.ctrl", "Key.shift", "h"]):
            logger.info("快捷键: 切换悬浮窗显示")
            self.callbacks.get("toggle_display", lambda: None)()

        # Ctrl+Shift+D → 切换调试模式
        if self._is_pressed(["Key.ctrl", "Key.shift", "d"]):
            logger.info("快捷键: 切换调试模式")
            self.callbacks.get("toggle_debug", lambda: None)()

        # Ctrl+Shift+P → 暂停/恢复
        if self._is_pressed(["Key.ctrl", "Key.shift", "p"]):
            logger.info("快捷键: 暂停/恢复")
            self.callbacks.get("toggle_pause", lambda: None)()

        # Ctrl+Q → 退出
        if self._is_pressed(["Key.ctrl", "q"]):
            logger.info("快捷键: 退出程序")
            self.callbacks.get("quit", lambda: None)()

    def _is_pressed(self, keys: list) -> bool:
        """检查指定组合键是否全部按下"""
        # 把 ctrl_l/ctrl_r 统一为 Key.ctrl
        normalized = set()
        for k in self._pressed_keys:
            if k in ("Key.ctrl_l", "Key.ctrl_r"):
                normalized.add("Key.ctrl")
            elif k in ("Key.shift_l", "Key.shift_r"):
                normalized.add("Key.shift")
            else:
                normalized.add(k)
        return all(k in normalized for k in keys)

    def stop(self):
        """停止监听"""
        self._running = False
        if self._listener:
            self._listener.stop()


# ==================== 主程序 ====================

class DecisionSystem:
    """王者荣耀智能决策系统主控制器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化决策系统

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.paused = False
        self.debug_mode = False
        self.running = False

        # 统计
        self.frame_count = 0
        self.detection_count = 0
        self.api_call_count = 0
        self.start_time = 0.0
        self.fps_history = []

        # 跳帧缓存
        self._last_frame = None          # 上一帧截图，用于判断画面变化
        self._last_detections = []       # 上次检测结果缓存
        self._last_positions = []        # 上次位置数据缓存

        logger.info("=" * 50)
        logger.info("王者荣耀智能决策系统 v1.0")
        logger.info("=" * 50)

    def initialize(self, skip_selector: bool = False) -> None:
        """
        初始化各模块

        Args:
            skip_selector: 跳过截取框调节界面
        """
        # 1. 屏幕截取器
        logger.info("初始化屏幕截取器...")
        self.capture = ScreenCapture(self.config_path)

        # 2. YOLO检测器
        logger.info("初始化YOLO检测器...")
        self.detector = YOLODetector.from_config(self.config_path)
        # 尝试预加载模型
        try:
            self.detector.load_model()
            logger.info("YOLO模型加载成功 ✓")
        except FileNotFoundError as e:
            logger.warning(f"模型文件未找到: {e}")
            logger.warning("系统将以模拟模式运行（跳过检测）")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")
            logger.warning("系统将以模拟模式运行（跳过检测）")

        # 3. API决策器
        logger.info("初始化API决策器...")
        self.decision_maker = APIDecisionMaker.from_config(self.config_path)

        # 4. 结果显示
        logger.info("初始化结果显示...")
        self.display = DecisionDisplay.from_config(self.config_path)
        self.display.start()

        # 5. 快捷键监听
        logger.info("初始化快捷键...")
        self.hotkeys = HotkeyListener({
            "reopen_selector": lambda: self.capture.open_selector(),
            "toggle_display": lambda: self.display.toggle(),
            "toggle_debug": self._toggle_debug,
            "toggle_pause": self._toggle_pause,
            "quit": self.shutdown,
        })
        self.hotkeys.start()

        logger.info("所有模块初始化完成 ✓")

        # 显示启动就绪状态，取代"等待战局分析..."
        self.display.show_decision({
            "action": "等待",
            "target": "战局分析",
            "reason": "系统就绪，等待截取区域确认",
            "priority": "low",
        })

    def _toggle_debug(self) -> None:
        """切换调试模式"""
        self.debug_mode = not self.debug_mode
        logger.info(f"调试模式: {'开启' if self.debug_mode else '关闭'}")

    def _toggle_pause(self) -> None:
        """切换暂停状态"""
        self.paused = not self.paused
        status = "暂停" if self.paused else "恢复"
        logger.info(f"系统{status}")
        self.display.show_decision({
            "action": status,
            "target": "" if self.paused else "继续分析",
            "reason": "用户手动暂停" if self.paused else "恢复正常监控",
            "priority": "medium",
        })

    def shutdown(self) -> None:
        """关闭系统"""
        logger.info("正在关闭系统...")
        self.running = False

    def run(self, test_capture: bool = False) -> None:
        """
        运行主循环

        Args:
            test_capture: 仅测试截图模式
        """
        self.running = True
        self.start_time = time.time()

        logger.info("主循环开始运行")
        logger.info("快捷键提示:")
        logger.info("  Ctrl+Shift+S → 调节截取框")
        logger.info("  Ctrl+Shift+H → 显示/隐藏悬浮窗")
        logger.info("  Ctrl+Shift+D → 切换调试模式")
        logger.info("  Ctrl+Shift+P → 暂停/恢复")
        logger.info("  Ctrl+Q     → 退出程序")

        frame_time = 0.0
        last_fps_print = time.time()

        try:
            while self.running:
                loop_start = time.time()

                # === 暂停状态 ===
                if self.paused:
                    time.sleep(0.1)
                    continue

                # === 1. 屏幕截取 ===
                try:
                    frame = self.capture.capture()
                except Exception as e:
                    logger.error(f"截图失败: {e}")
                    time.sleep(0.5)
                    continue

                if test_capture:
                    # 仅测试截图模式：显示截图（带区域信息）
                    info_frame = frame.copy()
                    region = self.capture.get_region()
                    h, w = info_frame.shape[:2]
                    cv2.putText(
                        info_frame,
                        f"区域: ({region['left']}, {region['top']}) {w}x{h}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
                    cv2.imshow(f"截图测试 - {w}x{h} (按Q退出)", info_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                # === 2. YOLO检测（智能跳帧：画面无变化时跳过） ===
                detections = []
                hero_positions = []

                if self.detector._model is not None:
                    # 计算与上一帧的差异，判断是否值得重新检测
                    need_detect = True
                    if self._last_frame is not None and frame.shape == self._last_frame.shape:
                        diff = cv2.absdiff(frame, self._last_frame)
                        mean_diff = diff.mean()
                        need_detect = mean_diff > 2.0  # 阈值：像素平均变化>2则触发检测

                    # 强制兜底：至少每20帧检测一次，防止长时间漏掉变化
                    if self.frame_count % 20 == 0:
                        need_detect = True

                    if need_detect:
                        try:
                            detections = self.detector.detect(frame)
                            hero_positions = self.detector.get_position_summary(detections)
                            self.detection_count += len(detections)
                            # 缓存本次结果
                            self._last_detections = detections
                            self._last_positions = hero_positions
                            self._last_frame = frame.copy()
                        except Exception as e:
                            logger.error(f"检测失败: {e}")
                            hero_positions = self._last_positions[:] if self._last_positions else []
                    else:
                        # 画面变化小：复用上次检测结果
                        detections = list(self._last_detections) if self._last_detections else []
                        hero_positions = list(self._last_positions) if self._last_positions else []
                else:
                    # 模型未加载时使用模拟数据（用于界面调试）
                    hero_positions = self._get_mock_positions()

                # === 3. 调试可视化 ===
                if self.debug_mode:
                    if detections:
                        debug_frame = self.detector.draw_detections(frame, detections)
                    else:
                        debug_frame = frame.copy()
                    # 在图上叠加信息
                    region = self.capture.get_region()
                    info_lines = [
                        f"截取区域: ({region['left']}, {region['top']}) {region['width']}x{region['height']}",
                        f"帧尺寸: {frame.shape[1]}x{frame.shape[0]}",
                        f"FPS: {self.fps_history[-1]:.1f}" if self.fps_history else "FPS: --",
                        f"检测: {len(detections)} 英雄",
                        f"API调用: {self.api_call_count}",
                    ]
                    for i, line in enumerate(info_lines):
                        cv2.putText(
                            debug_frame, line,
                            (5, 15 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 0), 1,
                        )

                    cv2.imshow("调试 - 检测可视化 (按Q退出)", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # === 4. API决策 ===
                if hero_positions:
                    try:
                        decision = self.decision_maker.get_decision(hero_positions)
                        self.api_call_count += 1

                        # 显示决策
                        self.display.show_decision(decision)
                    except Exception as e:
                        logger.error(f"决策获取失败: {e}")
                else:
                    # 无检测结果时显示空状态
                    if self.frame_count % 30 == 0:
                        self.display.show_decision({
                            "action": "观望",
                            "target": "无视野",
                            "reason": "未检测到英雄，等待视野",
                            "priority": "low",
                        })

                # === 5. FPS控制与统计 ===
                self.frame_count += 1
                elapsed = time.time() - loop_start
                target_interval = self.capture.frame_interval

                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

                # 计算实际FPS
                actual_frame_time = time.time() - loop_start
                current_fps = 1.0 / max(actual_frame_time, 0.001)
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 50:
                    self.fps_history.pop(0)

                # 每5秒打印一次状态
                if time.time() - last_fps_print >= 5:
                    avg_fps = sum(self.fps_history) / max(len(self.fps_history), 1)
                    total_time = time.time() - self.start_time
                    logger.info(
                        f"状态: FPS={avg_fps:.1f} | "
                        f"检测={self.detection_count} | "
                        f"API调用={self.api_call_count} | "
                        f"运行={total_time:.0f}s"
                    )
                    last_fps_print = time.time()

        except KeyboardInterrupt:
            logger.info("用户中断 (Ctrl+C)")

        finally:
            self._cleanup()

    def _get_mock_positions(self) -> list:
        """
        生成模拟英雄位置数据（用于模型未加载时的界面调试）

        Returns:
            模拟的英雄位置列表
        """
        import random

        # 每5帧才更新一次模拟位置，模拟真实的更新频率
        if self.frame_count % 5 != 0 and hasattr(self, "_mock_cache"):
            return self._mock_cache

        mock = [
            {"name": "libai", "team": "ally", "position": [random.uniform(0.2, 0.5), random.uniform(0.3, 0.7)]},
            {"name": "diaochan", "team": "ally", "position": [random.uniform(0.3, 0.6), random.uniform(0.3, 0.7)]},
            {"name": "hanxin", "team": "ally", "position": [random.uniform(0.4, 0.7), random.uniform(0.2, 0.6)]},
            {"name": "luna", "team": "ally", "position": [random.uniform(0.2, 0.5), random.uniform(0.4, 0.8)]},
            {"name": "ake", "team": "ally", "position": [random.uniform(0.3, 0.6), random.uniform(0.5, 0.9)]},
            {"name": "libai", "team": "enemy", "position": [random.uniform(0.5, 0.8), random.uniform(0.2, 0.6)]},
            {"name": "diaochan", "team": "enemy", "position": [random.uniform(0.5, 0.8), random.uniform(0.3, 0.7)]},
            {"name": "hanxin", "team": "enemy", "position": [random.uniform(0.4, 0.7), random.uniform(0.4, 0.8)]},
            {"name": "luna", "team": "enemy", "position": [random.uniform(0.6, 0.9), random.uniform(0.2, 0.5)]},
            {"name": "ake", "team": "enemy", "position": [random.uniform(0.5, 0.8), random.uniform(0.5, 0.9)]},
        ]
        self._mock_cache = mock
        return mock

    def _cleanup(self) -> None:
        """清理资源"""
        logger.info("正在清理资源...")

        # 停止快捷键监听
        if hasattr(self, "hotkeys"):
            self.hotkeys.stop()

        # 关闭显示器
        if hasattr(self, "display"):
            self.display.stop()

        # 关闭OpenCV窗口
        cv2.destroyAllWindows()

        # 统计摘要
        if self.start_time > 0:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / max(total_time, 0.1)
            logger.info("=" * 50)
            logger.info("会话统计:")
            logger.info(f"  运行时间: {total_time:.1f}s")
            logger.info(f"  处理帧数: {self.frame_count}")
            logger.info(f"  平均FPS: {avg_fps:.1f}")
            logger.info(f"  检测次数: {self.detection_count}")
            logger.info(f"  API调用: {self.api_call_count}")
            logger.info("=" * 50)

        logger.info("系统已关闭")


# ==================== CLI入口 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="王者荣耀智能决策系统 v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/main.py                          # 标准启动
  python src/main.py --config config/config.yaml  # 指定配置文件
  python src/main.py --debug                  # 调试模式
  python src/main.py --test-capture           # 仅测试截图
  python src/main.py --no-selector            # 跳过截取框调节
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="配置文件路径 (默认: config/config.yaml)",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="调试模式（显示检测可视化窗口）",
    )
    parser.add_argument(
        "--test-capture",
        action="store_true",
        help="仅测试屏幕截取功能",
    )
    parser.add_argument(
        "--no-selector",
        action="store_true",
        help="跳过截取框调节界面（使用已保存坐标）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 配置日志
    setup_logging(args.log_level)

    # 检查配置文件
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        logger.info("请确保配置文件路径正确，或从模板创建")
        sys.exit(1)

    # 创建并初始化系统
    system = DecisionSystem(args.config)

    try:
        # 初始化
        system.initialize(skip_selector=args.no_selector)

        # 调试模式
        if args.debug:
            system._toggle_debug()

        # 首次运行或测试截图：打开截取框调节
        if not args.no_selector and not args.test_capture:
            logger.info("打开截取框调节界面...")
            system.capture.open_selector()

        if args.test_capture:
            logger.info("进入截图测试模式，按 Ctrl+C 退出")
            system.display.show_decision({
                "action": "测试模式",
                "target": "屏幕截取",
                "reason": "正在测试截图功能，Ctrl+C退出",
                "priority": "medium",
            })

        # 运行主循环
        system.run(test_capture=args.test_capture)

    except Exception as e:
        logger.exception(f"系统运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
