# 王者荣耀智能决策系统 \- 项目README

> 部分内容由豆包生成
> 
> 

# 一、项目概述

**项目定位：**王者荣耀游戏智能决策辅助系统（Honor of Kings AI Decision Assistant）**核心技术：**YOLO目标检测 \+ 大语言模型API \+ 屏幕实时截取**应用场景：**实时战局分析、打野路线规划、团战时机判断、Gank预警本项目是一套基于计算机视觉与大语言模型的王者荣耀游戏自动化决策辅助系统。系统通过以下核心链路实现智能决策：**屏幕截取**：可手动调节截取框，精准捕获游戏小地图区域**视觉识别**：加载训练好的YOLO模型权重，检测并分类小地图上的敌我英雄头像**位置提取**：将识别结果转换为结构化的英雄坐标信息**决策推理**：将位置信息喂给大模型API，结合系统提示词输出当前时间帧的战术决策**结果展示**：以简洁醒目的方式在屏幕上显示决策建议系统设计注重可扩展性与易用性，所有关键参数均可通过配置文件调整，截取框支持运行时手动拖拽调整。

# 二、系统架构与工作流程

2\.1 整体架构图2\.2 工作数据流阶段输入处理输出屏幕采集电脑屏幕画面按截取框坐标裁剪小地图区域小地图截图（numpy数组）目标检测小地图截图YOLO模型推理 \+ NMS后处理英雄检测框列表（类别\+置信度\+坐标）信息提取检测结果坐标归一化 \+ 敌我阵营分类结构化JSON：英雄名\+相对位置决策推理位置JSON \+ 系统提示词调用大模型API进行战局分析当前决策建议文本结果展示决策文本格式化 \+ 悬浮窗渲染屏幕上简洁显示决策2\.3 项目目录结构hok\-ai\-decision/
├── config/
│   └── config\.yaml          \# 主配置文件（截取框、模型路径、API密钥等）
├── models/
│   └── best\.pt              \# 训练好的YOLO模型权重文件
├── src/
│   ├── screen\_capture\.py    \# 屏幕截取模块（含可调节截取框）
│   ├── yolo\_detector\.py     \# YOLO英雄识别模块
│   ├── api\_decision\.py      \# API大模型决策模块
│   ├── display\.py           \# 决策结果显示模块
│   └── main\.py              \# 主程序入口
├── prompts/
│   └── system\_prompt\.txt    \# 大模型系统提示词
├── requirements\.txt         \# Python依赖清单
└── README\.md                \# 本说明文档


# 四、核心模块详解

## 4\.1 屏幕截取模块（可调节截取框）

功能说明屏幕截取模块负责从电脑屏幕中捕获指定区域的画面。核心特性是**支持运行时手动调节截取框的大小和位置**，用户可以通过鼠标拖拽来精准框选游戏小地图所在的区域。核心功能**可调节截取框：**启动后显示半透明选框，鼠标拖拽边框调整大小，拖拽中间移动位置**实时预览：**调节过程中实时显示截取区域的缩略预览**坐标持久化：**调整后的截取框坐标自动保存到配置文件，下次启动自动加载**高性能截图：**基于mss库实现毫秒级屏幕截取，满足实时性要求**多显示器支持：**自动检测显示器数量，支持指定截取屏幕使用方式from src\.screen\_capture import ScreenCapture

\# 初始化截图器
capture = ScreenCapture\(config\_path="config/config\.yaml"\)

\# 打开调节界面（首次运行或手动触发时调用）
capture\.open\_selector\(\)  \# 弹出可拖拽的截取框

\# 截取当前框选区域
frame = capture\.capture\(\)  \# 返回numpy数组 \(H, W, 3\) BGR格式

\# 获取截取框坐标
region = capture\.get\_region\(\)
\# 返回: \{"left": 100, "top": 800, "width": 200, "height": 200\}调节截取框操作说明操作效果鼠标左键按住框体中间拖拽移动整个截取框位置鼠标左键按住边框/四角拖拽调整截取框大小按 Enter 键确认当前截取区域并保存按 Esc 键取消调整，恢复上次保存的坐标按 R 键重置为默认小地图位置关键实现技术使用 mss 库进行高性能屏幕捕获，比PIL的ImageGrab快3\-5倍使用 tkinter 或 PyQt5 实现半透明可拖拽选框截取坐标存储在config\.yaml的 capture\.region 字段

## 4\.2 YOLO英雄识别模块

功能说明YOLO英雄识别模块加载用户训练好的YOLO模型权重，对截取的小地图截图进行目标检测，识别出地图上各个英雄头像的类别、置信度和像素坐标。核心功能**多英雄分类：**支持识别王者荣耀全部英雄头像，输出英雄名称**敌我区分：**根据颜色特征或类别标签自动区分己方与敌方英雄**坐标归一化：**将像素坐标转换为小地图相对坐标（0\~1范围），便于后续决策**置信度过滤：**可配置置信度阈值，过滤低质量检测结果**批量推理：**支持单帧与批量推理，GPU加速下可达100\+ FPS使用方式from src\.yolo\_detector import YOLODetector

\# 初始化检测器
detector = YOLODetector\(
    model\_path="models/best\.pt",
    conf\_threshold=0\.5,
    iou\_threshold=0\.45,
    device="cuda"  \# 或 "cpu"
\)

\# 对单帧图片进行检测
results = detector\.detect\(frame\)  \# frame为numpy数组

\# 检测结果格式
print\(results\)
\# \[
\#   \{
\#     "class\_id": 3,
\#     "class\_name": "libai",     \# 英雄类别名
\#     "confidence": 0\.92,        \# 置信度
\#     "bbox": \[x1, y1, x2, y2\],  \# 像素坐标
\#     "norm\_pos": \[0\.35, 0\.72\],  \# 归一化坐标 \(相对于小地图\)
\#     "team": "enemy"            \# 阵营: ally / enemy
\#   \},
\#   \.\.\.
\# \]

\# 获取检测到的英雄位置摘要（喂给API的结构化数据）
position\_summary = detector\.get\_position\_summary\(results\)
\# 返回结构化字典，可直接序列化为JSON模型训练要求**数据集建议：**采集不同分辨率、不同画质下的小地图截图每个英雄样本数≥200张，包含不同朝向与缩放状态标注格式使用YOLO标准格式（class\_id x\_center y\_center width height）类别命名建议使用英雄拼音英文名，如：libai, diaochan, hanxin类别映射配置在 config/config\.yaml 中配置英雄类别ID到名称的映射：yolo:
  class\_names:
    0: "allied\_libai"
    1: "allied\_diaochan"
    2: "enemy\_libai"
    3: "enemy\_diaochan"
    \# \.\.\. 更多英雄
  \# 或使用颜色区分敌我
  team\_detection:
    enabled: true
    ally\_color: \[0, 100, 255\]   \# 己方蓝色调
    enemy\_color: \[255, 50, 50\]  \# 敌方红色调性能优化建议使用 model\.eval\(\) \+ torch\.no\_grad\(\) 减少显存占用小地图输入尺寸建议设置为 640×640 或 416×416，平衡速度与精度启用 half=True 半精度推理，速度提升约40%使用TensorRT或ONNX Runtime进一步加速部署

## 4\.3 API决策推理模块

功能说明API决策模块负责将YOLO识别出的英雄位置信息进行结构化封装，结合预设的系统提示词调用大语言模型API，让AI根据当前战局态势输出具体的战术决策建议。核心功能**多API提供商支持：**兼容OpenAI、豆包、通义千问、DeepSeek等主流大模型API**结构化输入：**自动将检测结果转换为模型易理解的JSON格式**系统提示词管理：**独立的prompt文件，可自定义AI的角色设定与决策规则**速率限制控制：**内置请求频率控制，避免触发API限流**缓存机制：**位置变化不大时复用上次决策结果，减少API调用次数**错误重试：**网络异常时自动重试，保障系统稳定性使用方式from src\.api\_decision import APIDecisionMaker

\# 初始化决策器
decision\_maker = APIDecisionMaker\(config\_path="config/config\.yaml"\)

\# 传入英雄位置信息，获取决策
hero\_positions = \[
    \{"name": "李白", "team": "ally", "position": \[0\.35, 0\.72\]\},
    \{"name": "貂蝉", "team": "ally", "position": \[0\.5, 0\.5\]\},
    \{"name": "韩信", "team": "enemy", "position": \[0\.8, 0\.3\]\},
    \# \.\.\.
\]

decision = decision\_maker\.get\_decision\(hero\_positions\)

\# 返回决策结果
print\(decision\)
\# \{
\#   "action": "撤退",
\#   "target": "我方蓝buff区域",
\#   "reason": "敌方韩信出现在我方野区，人数劣势，建议暂避锋芒",
\#   "priority": "high",
\#   "timestamp": 1234567890
\# \}系统提示词设计系统提示词位于 prompts/system\_prompt\.txt，可根据需求自定义AI的决策风格。示例模板：你是一名王者荣耀职业级辅助决策AI，职责是根据小地图英雄位置给出最优战术建议。

【输入格式】
你将收到一个JSON数组，每个元素包含英雄名称、阵营（ally我方/enemy敌方）、归一化坐标\[x, y\]。
坐标范围0\~1，\(0,0\)为左上角，\(1,1\)为右下角，对应小地图范围。

【输出要求】
严格输出JSON格式，包含以下字段：
\- action: 建议动作（进攻/撤退/发育/打龙/推塔/支援/蹲守）
\- target: 目标地点（具体到某条路/某个野区/某个塔）
\- reason: 简短理由（不超过30字）
\- priority: 优先级（high/medium/low）

【决策原则】
1\. 人数占优时主动找机会开团
2\. 敌方关键英雄消失时提示警惕
3\. 优先保护核心C位
4\. 龙刷新时根据局势判断是否争夺

只输出JSON，不要多余解释。支持的API提供商提供商配置值推荐模型备注OpenAIopenaigpt\-4o\-mini / gpt\-3\.5\-turbo通用兼容格式豆包（字节）doubaodoubao\-pro\-32k国产模型，响应快通义千问qwenqwen\-plus阿里出品DeepSeekdeepseekdeepseek\-chat性价比高自定义custom\-兼容OpenAI格式的任意接口调用频率优化**位置变化阈值：**只有当英雄平均移动距离超过阈值时才触发新的API调用**最小调用间隔：**配置 api\.min\_interval 控制每秒最大请求数**决策缓存：**相同态势下直接返回缓存结果，节省token消耗

## 4\.4 决策结果显示模块

功能说明决策结果显示模块负责将大模型输出的决策建议以**简洁、醒目、不遮挡游戏关键区域**的方式展示在屏幕上，确保玩家能快速获取当前战术指令。核心功能**悬浮窗显示：**无边框半透明悬浮窗口，始终置顶不影响游戏操作**简洁输出：**只显示核心动作与目标，避免信息过载**优先级配色：**高优先级红色警示、中优先级黄色提醒、低优先级绿色提示**位置可配置：**支持设置悬浮窗在屏幕的位置（左上/右上/左下/右下/自定义坐标）**自动淡出：**新决策高亮显示，几秒后自动降低透明度**历史记录：**可展开查看最近5条决策历史显示效果示例**【高优先级】撤退 → 我方蓝buff区**敌方韩信入侵野区，暂避锋芒**【中优先级】支援 → 中路**中路即将爆发团战，速往支援**【低优先级】发育 → 下路带线**局势平稳，优先发育经济使用方式from src\.display import DecisionDisplay

\# 初始化显示器
display = DecisionDisplay\(
    position="top\-right",  \# 屏幕右上角
    opacity=0\.9,
    font\_size=16,
    show\_reason=True
\)

\# 显示一条决策
decision = \{
    "action": "撤退",
    "target": "我方蓝buff区域",
    "reason": "敌方韩信出现在我方野区",
    "priority": "high"
\}
display\.show\_decision\(decision\)

\# 更新决策（自动替换当前显示）
display\.update\_decision\(new\_decision\)

\# 隐藏/显示窗口
display\.hide\(\)
display\.show\(\)

\# 关闭显示
display\.close\(\)显示样式配置display:
  \# 窗口位置: top\-left / top\-right / bottom\-left / bottom\-right / custom
  position: "top\-right"
  \# 自定义坐标（position为custom时生效）
  custom\_x: 100
  custom\_y: 100
  \# 窗口透明度 0\.0\~1\.0
  opacity: 0\.85
  \# 字体大小
  font\_size: 18
  \# 是否显示理由
  show\_reason: true
  \# 高亮持续时间（秒）
  highlight\_duration: 3
  \# 优先级颜色配置
  colors:
    high: "\#ff4444"
    medium: "\#ffaa00"
    low: "\#44dd44"
  \# 窗口宽度（像素）
  width: 280技术实现方案**方案A（推荐）：**使用 PyQt5 实现无边框置顶窗口，渲染效果好，支持丰富样式**方案B：**使用 tkinter \+ attributes\("\-topmost", True\)，轻量无额外依赖**方案C：**使用 pygame 绘制透明覆盖层，适合需要动画效果的场景**优化建议：**悬浮窗应避开游戏小地图、技能按钮等关键操作区域，建议默认放在屏幕顶部或侧边。可在设置中手动拖拽调整悬浮窗位置。

# 五、快速开始（使用教程）

5\.1 首次运行完整流程**启动游戏**：打开王者荣耀，进入对局，确保小地图正常显示**运行主程序**：执行启动命令**调节截取框**：首次运行会弹出截取框调节界面，框选游戏小地图区域**确认区域**：按 Enter 键确认截取区域，系统开始自动识别**查看决策**：屏幕上会显示实时决策建议5\.2 启动命令\# 标准启动（自动加载配置）
python src/main\.py

\# 指定配置文件启动
python src/main\.py \-\-config config/config\.yaml

\# 调试模式（显示检测可视化窗口）
python src/main\.py \-\-debug

\# 仅测试屏幕截取
python src/main\.py \-\-test\-capture

\# 跳过截取框调节（使用上次保存的坐标）
python src/main\.py \-\-no\-selector5\.3 运行时快捷键快捷键功能Ctrl \+ Shift \+ S重新打开截取框调节界面Ctrl \+ Shift \+ H显示/隐藏决策悬浮窗Ctrl \+ Shift \+ D切换调试模式（显示检测画面）Ctrl \+ Shift \+ P暂停/恢复识别Ctrl \+ Q退出程序5\.4 主程序入口示例import time
from src\.screen\_capture import ScreenCapture
from src\.yolo\_detector import YOLODetector
from src\.api\_decision import APIDecisionMaker
from src\.display import DecisionDisplay

def main\(\):
    \# 1\. 初始化各模块
    capture = ScreenCapture\("config/config\.yaml"\)
    detector = YOLODetector\("models/best\.pt"\)
    decision\_maker = APIDecisionMaker\("config/config\.yaml"\)
    display = DecisionDisplay\(\)

    \# 2\. 首次运行调节截取框
    capture\.open\_selector\(\)

    last\_api\_call = 0
    min\_interval = 2\.0  \# API最小调用间隔（秒）

    \# 3\. 主循环
    while True:
        \# 截取屏幕
        frame = capture\.capture\(\)

        \# YOLO识别英雄
        detections = detector\.detect\(frame\)
        hero\_positions = detector\.get\_position\_summary\(detections\)

        \# 控制API调用频率
        current\_time = time\.time\(\)
        if current\_time \- last\_api\_call \>= min\_interval:
            \# 获取AI决策
            decision = decision\_maker\.get\_decision\(hero\_positions\)
            display\.show\_decision\(decision\)
            last\_api\_call = current\_time

        time\.sleep\(0\.1\)  \# 控制主循环频率

if \_\_name\_\_ == "\_\_main\_\_":
    main\(\)

# 六、配置文件说明

6\.1 完整配置文件示例\# ============== 屏幕截取配置 ==============
capture:
  \# 截取区域坐标（首次运行后自动保存）
  region:
    left: 0
    top: 800
    width: 220
    height: 220
  \# 目标显示器（多显示器时指定，0为主屏）
  monitor: 0
  \# 截图帧率（每秒截取次数）
  fps: 10

\# ============== YOLO模型配置 ==============
yolo:
  \# 模型权重路径
  model\_path: "models/best\.pt"
  \# 置信度阈值
  conf\_threshold: 0\.5
  \# IOU阈值（NMS用）
  iou\_threshold: 0\.45
  \# 推理设备: cuda / cpu
  device: "cuda"
  \# 输入图像尺寸
  input\_size: 640
  \# 半精度推理
  half\_precision: true
  \# 英雄类别名称映射
  class\_names:
    0: "ally\_libai"
    1: "ally\_diaochan"
    2: "ally\_hanxin"
    3: "enemy\_libai"
    4: "enemy\_diaochan"
    5: "enemy\_hanxin"
    \# \.\.\. 根据训练的类别补充

\# ============== API决策配置 ==============
api:
  \# API提供商: openai / doubao / qwen / deepseek / custom
  provider: "doubao"
  \# API密钥
  api\_key: "your\-api\-key\-here"
  \# API基础地址
  base\_url: "https://ark\.cn\-beijing\.volces\.com/api/v3"
  \# 使用的模型名称
  model: "doubao\-pro\-32k"
  \# 温度参数（0\~1，越低越稳定）
  temperature: 0\.3
  \# 最大token数
  max\_tokens: 200
  \# 最小调用间隔（秒）
  min\_interval: 2\.0
  \# 系统提示词文件路径
  system\_prompt\_path: "prompts/system\_prompt\.txt"
  \# 重试次数
  max\_retries: 3
  \# 超时时间（秒）
  timeout: 10

\# ============== 结果显示配置 ==============
display:
  \# 是否启用显示
  enabled: true
  \# 窗口位置
  position: "top\-right"
  \# 透明度 0\.0\~1\.0
  opacity: 0\.85
  \# 字体大小
  font\_size: 18
  \# 是否显示理由
  show\_reason: true
  \# 高亮持续时间（秒）
  highlight\_duration: 3
  \# 窗口宽度
  width: 300

\# ============== 调试配置 ==============
debug:
  \# 是否显示检测可视化窗口
  show\_detection\_window: false
  \# 是否保存检测截图
  save\_detections: false
  \# 截图保存目录
  save\_dir: "debug\_screenshots"
  \# 日志级别: DEBUG / INFO / WARNING / ERROR
  log\_level: "INFO"6\.2 关键参数调优指南参数调大的影响调小的影响推荐值conf\_threshold误报减少，漏检增加漏检减少，误报增加0\.4 \~ 0\.6api\.min\_intervalAPI消耗减少，响应延迟响应更快，API消耗增加1\.5 \~ 3\.0 秒capture\.fps识别更及时，CPU占用高资源占用低，有延迟5 \~ 15 FPSyolo\.input\_size精度提升，速度下降速度更快，精度下降416 / 640

# 七、常见问题与排错

7\.1 安装与环境问题**Q: 提示 "No module named ultralytics"**A: 执行 pip install ultralytics 安装YOLO依赖库。如使用YOLOv5则安装 pip install yolov5。**Q: CUDA不可用，模型跑在CPU上很慢**A: 检查是否安装了GPU版本的PyTorch。执行 python \-c "import torch; print\(torch\.cuda\.is\_available\(\)\)" 验证。返回False说明需要重装对应CUDA版本的PyTorch。**Q: mss库截图黑屏或截不到内容**A: Windows下尝试以管理员身份运行程序；游戏使用全屏模式时可能被保护，建议改用窗口化或无边框窗口模式。7\.2 识别效果问题**Q: YOLO识别准确率低，经常认错英雄**A: ① 检查训练数据集是否充足，每个英雄建议≥200样本；② 调低conf\_threshold看是否漏检；③ 确认游戏内画质与训练集一致；④ 小地图缩放比例变化大时建议补充不同缩放的训练数据。**Q: 同一个英雄被检测出多个框**A: 调大iou\_threshold参数（如从0\.45调到0\.6），增强NMS抑制效果。**Q: 敌我阵营区分不准确**A: 建议训练时直接区分类别（如ally\_libai和enemy\_libai作为两个类别），而不是靠颜色后处理判断。小地图上英雄头像边框颜色是区分敌我的关键特征。7\.3 API调用问题**Q: API返回格式不是JSON，解析失败**A: 在系统提示词中强调"只输出JSON，不要任何额外文字"；同时代码中增加容错解析，使用正则提取JSON部分。**Q: API调用频繁触发限流**A: 调大api\.min\_interval参数，增加最小调用间隔；启用位置变化阈值判断，只有局势变化时才调用。**Q: 网络超时导致程序卡顿**A: 配置合理的timeout参数；API调用放在独立线程中执行，不要阻塞主循环；增加本地缓存机制。7\.4 显示与交互问题**Q: 悬浮窗被游戏窗口挡住，看不到**A: 确保设置了窗口置顶属性；部分全屏独占模式的游戏会覆盖所有窗口，改用无边框窗口模式即可。**Q: 截取框调节界面无法拖拽**A: 检查是否有其他程序抢占了鼠标焦点；Linux下需要安装相应的GUI依赖库。7\.5 性能优化建议**提升识别速度：**使用GPU推理、开启半精度、减小input\_size、减少检测类别数**降低CPU占用：**降低截图帧率、使用mss替代PIL截图、API调用异步化**节省API费用：**增大调用间隔、增加位置变化阈值、启用决策缓存复用**提升决策质量：**优化系统提示词、增加历史状态上下文、补充更多战局规则王者荣耀智能决策系统 v1\.0 \| 技术文档

> （注：部分内容可能由 AI 生成）
