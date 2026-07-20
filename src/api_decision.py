"""
API决策推理模块 - 调用大语言模型进行战术决策

功能：
- 多API提供商支持（OpenAI / 豆包 / 通义千问 / DeepSeek / 自定义）
- 结构化输入组装
- 系统提示词管理
- 速率限制与缓存
- 错误重试与容错
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ==================== API提供商配置表 ====================

PROVIDER_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-pro-32k",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-flash",
    },
}


class APIDecisionMaker:
    """大模型API决策器"""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 200,
        min_interval: float = 2.0,
        max_retries: int = 3,
        timeout: int = 10,
        system_prompt_path: str = "prompts/system_prompt.txt",
        position_change_threshold: float = 0.05,
        enable_cache: bool = True,
    ):
        """
        初始化API决策器

        Args:
            provider: API提供商名称
            api_key: API密钥
            base_url: API基础地址
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大输出token
            min_interval: 最小调用间隔（秒）
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            system_prompt_path: 系统提示词文件路径
            position_change_threshold: 位置变化阈值
            enable_cache: 是否启用缓存
        """
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.system_prompt_path = system_prompt_path
        self.position_change_threshold = position_change_threshold
        self.enable_cache = enable_cache

        # 确定 base_url 和 model
        if provider in PROVIDER_CONFIGS:
            pc = PROVIDER_CONFIGS[provider]
            self.base_url = base_url or pc["base_url"]
            self.model = model or pc["default_model"]
        else:
            # 自定义提供商
            self.base_url = base_url
            self.model = model

        # 加载系统提示词
        self.system_prompt = self._load_system_prompt()

        # 状态管理
        self._last_call_time = 0.0
        self._last_positions: Optional[List[Dict]] = None
        self._cached_decision: Optional[Dict] = None
        self._client = None

        logger.info(
            f"API决策器初始化: provider={provider}, model={self.model}, "
            f"interval={min_interval}s"
        )

    @classmethod
    def from_config(cls, config_path: str) -> "APIDecisionMaker":
        """
        从配置文件创建决策器实例

        Args:
            config_path: 配置文件路径

        Returns:
            APIDecisionMaker实例
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"加载配置文件失败: {e}")
            config = {}

        api_cfg = config.get("api", {})

        # 支持环境变量解析 ${VAR_NAME}
        api_key = api_cfg.get("api_key", "")
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")
            if not api_key:
                logger.warning(f"环境变量 {env_var} 未设置，API密钥为空")

        return cls(
            provider=api_cfg.get("provider", "openai"),
            api_key=api_key,
            base_url=api_cfg.get("base_url", ""),
            model=api_cfg.get("model", ""),
            temperature=api_cfg.get("temperature", 0.3),
            max_tokens=api_cfg.get("max_tokens", 200),
            min_interval=api_cfg.get("min_interval", 2.0),
            max_retries=api_cfg.get("max_retries", 3),
            timeout=api_cfg.get("timeout", 10),
            system_prompt_path=api_cfg.get(
                "system_prompt_path", "prompts/system_prompt.txt"
            ),
            position_change_threshold=api_cfg.get("position_change_threshold", 0.05),
            enable_cache=api_cfg.get("enable_cache", True),
        )

    def _load_system_prompt(self) -> str:
        """加载系统提示词文件"""
        try:
            with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            logger.info(f"系统提示词加载成功: {self.system_prompt_path}")
            return prompt
        except FileNotFoundError:
            logger.warning(
                f"系统提示词文件不存在: {self.system_prompt_path}，使用默认提示词"
            )
            return "你是一个游戏战术助手，请根据战场情况给出建议。只输出JSON。"
        except IOError as e:
            logger.error(f"读取系统提示词失败: {e}")
            return "你是一个游戏战术助手。"

    def _get_client(self):
        """获取或创建OpenAI客户端"""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
                logger.info(f"API客户端已创建: {self.base_url}")
            except ImportError:
                raise ImportError(
                    "未安装 openai 库，请执行: pip install openai"
                )
        return self._client

    # ==================== 核心决策接口 ====================

    def get_decision(self, hero_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        根据英雄位置信息获取战术决策

        Args:
            hero_positions: 英雄位置列表，格式:
                [
                    {"name": "libai", "team": "ally", "position": [0.35, 0.72]},
                    {"name": "hanxin", "team": "enemy", "position": [0.8, 0.3]},
                    ...
                ]

        Returns:
            决策字典:
            {
                "action": str,
                "target": str,
                "reason": str,
                "priority": "high" | "medium" | "low",
                "timestamp": float
            }
        """
        # 检查速率限制
        elapsed = time.time() - self._last_call_time
        if elapsed < self.min_interval:
            logger.debug(f"距离上次API调用仅 {elapsed:.1f}s，跳过（最小间隔{self.min_interval}s）")
            if self._cached_decision:
                return self._cached_decision
            return self._empty_decision("冷却中...")

        # 检查位置变化（缓存命中）
        if self.enable_cache and self._should_use_cache(hero_positions):
            logger.debug("英雄位置变化较小，使用缓存决策")
            if self._cached_decision:
                return self._cached_decision

        # 调用API
        decision = self._call_api(hero_positions)

        # 更新缓存
        self._last_call_time = time.time()
        self._last_positions = hero_positions
        self._cached_decision = decision

        return decision

    def _should_use_cache(self, hero_positions: List[Dict]) -> bool:
        """判断是否应该使用缓存的决策结果"""
        if not self._last_positions or not self._cached_decision:
            return False

        if len(hero_positions) != len(self._last_positions):
            return False

        # 计算英雄平均移动距离
        total_dist = 0.0
        for cur, prev in zip(hero_positions, self._last_positions):
            dx = cur["position"][0] - prev["position"][0]
            dy = cur["position"][1] - prev["position"][1]
            total_dist += (dx ** 2 + dy ** 2) ** 0.5

        avg_dist = total_dist / max(len(hero_positions), 1)
        return avg_dist < self.position_change_threshold

    def _call_api(self, hero_positions: List[Dict]) -> Dict[str, Any]:
        """调用大模型API"""
        # 构建用户消息
        user_message = json.dumps(hero_positions, ensure_ascii=False, indent=2)

        # === 打印最终传给模型的完整信息（DEBUG日志级别，加--debug或在config设log_level: DEBUG可见） ===
        # 如果想始终可见，下面三行取消注释：
        #print("\n========== 传给决策模型的完整信息 ==========")
        # print(f"系统提示词:\n{self.system_prompt}")
        #print(f"\n英雄位置:\n{user_message}")
        #print("==========================================\n")

        client = self._get_client()

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"API调用 (attempt {attempt + 1}/{self.max_retries + 1})")
                logger.debug(f"请求内容: {user_message[:200]}...")

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                content = response.choices[0].message.content
                logger.debug(f"API响应: {content}")

                # 解析JSON
                decision = self._parse_response(content)
                decision["timestamp"] = time.time()
                return decision

            except Exception as e:
                logger.warning(f"API调用失败 (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    wait = 2 ** attempt  # 指数退避
                    logger.info(f"等待 {wait}s 后重试...")
                    time.sleep(wait)
                else:
                    logger.error(f"API调用全部失败，返回空决策")
                    return self._empty_decision(f"API错误: {str(e)[:30]}")

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """
        解析API响应内容为决策字典

        容错策略：
        1. 直接解析整个内容
        2. 提取```json代码块
        3. 正则提取第一个JSON对象
        """
        # 策略1: 直接解析
        try:
            decision = json.loads(content)
            return self._validate_decision(decision)
        except json.JSONDecodeError:
            pass

        # 策略2: 提取 markdown 代码块
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group(1).strip())
                return self._validate_decision(decision)
            except json.JSONDecodeError:
                pass

        # 策略3: 正则提取JSON对象
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group(0))
                return self._validate_decision(decision)
            except json.JSONDecodeError:
                pass

        # 全部失败
        if content:
            logger.warning(f"API响应格式异常: {content[:300]}")
        else:
            logger.warning("API返回了空响应（可能API密钥未配置或网络异常）")
        return self._empty_decision("响应解析失败")

    def _validate_decision(self, decision: Dict) -> Dict[str, Any]:
        """验证并补全决策字段"""
        return {
            "action": decision.get("action", "观望"),
            "target": decision.get("target", "待定"),
            "reason": decision.get("reason", ""),
            "priority": decision.get("priority", "medium"),
            "suggestion": decision.get("suggestion", ""),
        }

    def _empty_decision(self, reason: str = "") -> Dict[str, Any]:
        """生成空的默认决策"""
        return {
            "action": "观望",
            "target": "待定",
            "reason": reason or "等待更多信息",
            "priority": "low",
            "timestamp": time.time(),
        }

    def clear_cache(self) -> None:
        """清除决策缓存"""
        self._cached_decision = None
        self._last_positions = None
        logger.debug("决策缓存已清除")
