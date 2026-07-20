"""
YOLO英雄识别模块 - 小地图英雄头像检测

功能：
- 加载训练好的YOLO模型权重
- 对小地图截图进行目标检测
- 识别英雄类别、置信度、像素坐标
- 区分敌我阵营
- 坐标归一化
- 生成结构化位置摘要
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO目标检测器，用于识别小地图上的英雄头像"""

    def __init__(
        self,
        model_path: str = "models/best.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        input_size: int = 640,
        half_precision: bool = True,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        初始化YOLO检测器

        Args:
            model_path: YOLO模型权重文件路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            device: 推理设备 ("cuda" / "cpu")
            input_size: 模型输入尺寸
            half_precision: 是否启用半精度推理
            class_names: 类别ID到名称的映射
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.input_size = input_size
        self.half_precision = half_precision
        self.class_names = class_names or {}

        # 延迟加载模型
        self._model = None
        self._frame_h = 0
        self._frame_w = 0

        logger.info(f"YOLO检测器初始化: model={model_path}, device={device}, conf={conf_threshold}")

    @classmethod
    def from_config(cls, config_path: str) -> "YOLODetector":
        """
        从配置文件创建检测器实例

        Args:
            config_path: 配置文件路径

        Returns:
            YOLODetector实例
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"加载配置文件失败: {e}")
            config = {}

        yolo_cfg = config.get("yolo", {})
        device = yolo_cfg.get("device", "cuda")

        # 自动检查CUDA可用性，不可用时降级到CPU
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA不可用，自动降级到CPU设备")
                    device = "cpu"
                else:
                    logger.info(f"CUDA可用，使用GPU推理")
            except ImportError:
                logger.warning("未安装torch，自动使用CPU设备")
                device = "cpu"

        return cls(
            model_path=yolo_cfg.get("model_path", "models/best.pt"),
            conf_threshold=yolo_cfg.get("conf_threshold", 0.5),
            iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
            device=device,
            input_size=yolo_cfg.get("input_size", 640),
            half_precision=yolo_cfg.get("half_precision", True) if device == "cuda" else False,
            class_names=yolo_cfg.get("class_names", {}),
        )

    def load_model(self) -> None:
        """加载YOLO模型（在使用前调用）"""
        if self._model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"模型权重文件不存在: {self.model_path}\n"
                "请将训练好的YOLO模型权重文件放置到 models/ 目录下"
            )

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)

            # 启用半精度（FP16）加速，仅在 CUDA 下生效
            if self.half_precision and self.device == "cuda":
                self._model = self._model.half()
                logger.info("已启用 FP16 半精度推理")

            logger.info(f"YOLO模型加载成功: {self.model_path}")
        except ImportError:
            raise ImportError(
                "未安装 ultralytics 库，请执行: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"YOLO模型加载失败: {e}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        对单帧图像进行目标检测

        Args:
            frame: 输入图像 (H, W, 3) BGR格式

        Returns:
            检测结果列表，每个元素包含:
            {
                "class_id": int,        # 类别ID
                "class_name": str,      # 类别名称
                "confidence": float,    # 置信度
                "bbox": [x1, y1, x2, y2],  # 像素坐标
                "norm_pos": [x, y],     # 归一化坐标 (相对小地图)
                "team": "ally" | "enemy" | "unknown"
            }
        """
        self.load_model()

        self._frame_h, self._frame_w = frame.shape[:2]

        # 执行推理
        results = self._model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        # 解析结果
        detections = self._parse_results(results)

        logger.debug(f"检测到 {len(detections)} 个英雄")
        return detections

    def debug_detect(self, frame: np.ndarray, save_dir: str = "debug_detections") -> dict:
        """
        诊断模式检测：打印完整的输入输出信息，保存中间图像

        Args:
            frame: 输入图像 (H, W, 3) BGR格式
            save_dir: 保存目录

        Returns:
            {"detections": [...], "frame_info": {...}, "model_input": {...}}
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        h, w = frame.shape[:2]
        info = {
            "frame_info": {
                "shape": f"{w}x{h}",
                "dtype": str(frame.dtype),
                "min_pixel": int(frame.min()),
                "max_pixel": int(frame.max()),
                "mean_pixel": f"{frame.mean():.1f}",
            },
            "model_input": {
                "imgsz": self.input_size,
                "device": self.device,
                "half": self.half_precision,
            },
        }

        # 保存原始帧
        import cv2
        orig_path = os.path.join(save_dir, "0_original.png")
        cv2.imwrite(orig_path, frame)
        print(f"\n{'='*60}")
        print(f"📸 YOLO 诊断信息")
        print(f"{'='*60}")
        print(f"输入帧尺寸: {w}x{h}")
        print(f"输入帧dtype: {frame.dtype}")
        print(f"像素范围: [{frame.min()}, {frame.max()}]")
        print(f"模型input_size: {self.input_size}")
        print(f"推理设备: {self.device}")
        print(f"原始帧已保存: {orig_path}")
        print(f"\n--- 推理预处理过程 ---")
        print(f"1. 输入帧: {w}x{h}")
        print(f"2. 等比例缩放 + 填充至: {self.input_size}x{self.input_size}")

        # 推理
        results = self._model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        # 用 ultralytics 内置方法保存预处理后的图
        for result in results:
            if hasattr(result, "orig_img"):
                resized_path = os.path.join(save_dir, "1_preprocessed.png")
                # 保存预处理后的图像（已经 letterbox 过的）
                if hasattr(result, "plot"):
                    plot_img = result.plot()
                    cv2.imwrite(resized_path, plot_img)
                    print(f"3. 检测可视化已保存: {resized_path}")

            if result.boxes is not None:
                print(f"\n--- 检测结果 ({len(result.boxes)} 个目标) ---")
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                for j in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[j].astype(int)
                    name = self.class_names.get(cls_ids[j], f"cls_{cls_ids[j]}")
                    print(f"  [{j}] {name:20s} conf={confs[j]:.3f}  "
                          f"bbox=({x1},{y1})-({x2},{y2})  "
                          f"size={x2-x1}×{y2-y1}px")
            else:
                print(f"\n--- 无检测结果 ---")

        # 解析标准结果
        detections = self._parse_results(results)
        print(f"\n--- 最终传递给 API 的结构化数据 ---")
        from pprint import pformat
        print(pformat(self.get_position_summary(detections)))
        print(f"{'='*60}\n")

        return {"detections": detections, **info}

    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """解析YOLO推理结果"""
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # 获取检测框坐标 (x1, y1, x2, y2)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy.astype(int).tolist()

                # 置信度
                confidence = float(boxes.conf[i].cpu().numpy())

                # 类别
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, f"hero_{class_id}")

                # 归一化坐标 (相对于小地图)
                center_x = (x1 + x2) / 2 / self._frame_w
                center_y = (y1 + y2) / 2 / self._frame_h
                norm_pos = [round(center_x, 4), round(center_y, 4)]

                # 判断阵营
                team = self._classify_team(class_name)

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2],
                    "norm_pos": norm_pos,
                    "team": team,
                })

        return detections

    def _classify_team(self, class_name: str) -> str:
        """
        根据类别名称判断阵营

        命名格式: {英雄名}_{颜色}
        - _blue  → ally（己方蓝色）
        - _red   → enemy（敌方红色）
        - _green → unknown（中立/未知）
        同时也向后兼容 ally_ / enemy_ 前缀格式
        """
        name_lower = class_name.lower()

        # 优先检查颜色后缀格式
        # _blue=队友(蓝方)  _green=本人(绿点)  → 都是己方 ally
        # _red=敌方(红方)  → enemy
        if name_lower.endswith("_blue") or name_lower.endswith("_green"):
            return "ally"
        elif name_lower.endswith("_red"):
            return "enemy"

        # 兼容旧格式：ally_前缀 → 己方
        if name_lower.startswith("ally_") or "ally" in name_lower:
            return "ally"
        # 兼容旧格式：enemy_前缀 → 敌方
        elif name_lower.startswith("enemy_") or "enemy" in name_lower:
            return "enemy"

        return "unknown"

    def _extract_hero_name(self, class_name: str) -> str:
        """从完整类别名中提取英雄名（去掉阵营/颜色后缀）"""
        name = class_name.lower()

        # 颜色后缀格式: zhugeliang_blue → zhugeliang
        for suffix in ["_blue", "_red", "_green"]:
            if name.endswith(suffix):
                return class_name[:-len(suffix)]

        # 兼容旧前缀格式: ally_libai → libai
        for prefix in ["ally_", "enemy_"]:
            if name.startswith(prefix):
                return class_name[len(prefix):]

        return class_name

    def get_position_summary(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将检测结果转换为结构化位置摘要（用于喂给API）

        Args:
            detections: detect() 返回的检测结果列表

        Returns:
            结构化摘要列表:
            [
                {"name": "libai", "team": "ally", "position": [0.35, 0.72]},
                ...
            ]
        """
        summary = []
        for det in detections:
            summary.append({
                "name": self._extract_hero_name(det["class_name"]),
                "team": det["team"],
                "position": det["norm_pos"],
            })
        return summary

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        在图像上绘制检测结果（用于调试可视化）

        Args:
            frame: 原始图像
            detections: 检测结果列表

        Returns:
            绘制了检测框的图像
        """
        vis = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            team = det["team"]

            # 按阵营选择颜色
            if team == "ally":
                color = (255, 100, 0)    # 蓝色(BGR) - 己方
            elif team == "enemy":
                color = (0, 50, 255)     # 红色(BGR) - 敌方
            else:
                color = (200, 200, 200)  # 灰色 - 未知

            # 绘制矩形框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{self._extract_hero_name(class_name)} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                vis,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0] + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                vis, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1,
            )

        return vis
