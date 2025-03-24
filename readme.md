```
anomaly_detection_system/
│
├── main.py                       # 主入口点脚本
├── config.py                     # 系统配置和常量
├── requirements.txt              # 依赖项列表
├── README.md                     # 项目文档
│
├── data/                         # 数据目录
│   ├── logon.csv
│   ├── device.csv
│   ├── email.csv
│   ├── file.csv
│   ├── http.csv
│   ├── LDAP.csv
│   └── psychometric.csv
│
├── anomaly_detection/            # 主包
│   │
│   ├── __init__.py               # 包初始化文件
│   │
│   ├── preprocessors/            # 预处理组件子包
│   │   ├── __init__.py
│   │   ├── preprocessing_coordinator.py   # 预处理协调器
│   │   └── data_loader.py        # 数据加载组件
│   │
│   ├── profile/                  # 用户配置文件相关组件
│   │   ├── __init__.py
│   │   └── user_profile_builder.py  # 用户配置文件构建器
│   │
│   ├── models/                   # 异常检测模型
│   │   ├── __init__.py
│   │   ├── base_detector.py      # 基础检测器接口
│   │   └── multidimensional_detector.py  # 多维异常检测器
│   │
│   ├── analyzers/                # 具体分析器
│   │   ├── __init__.py
│   │   ├── time_analyzer.py      # 时间异常分析器
│   │   ├── access_analyzer.py    # 访问异常分析器
│   │   ├── email_analyzer.py     # 邮件异常分析器
│   │   └── org_analyzer.py       # 组织异常分析器
│   │
│   ├── utils/                    # 工具函数子包
│   │   ├── __init__.py
│   │   ├── timestamp_processor.py  # 时间戳处理工具
│   │   ├── text_processor.py      # 文本处理工具
│   │   └── feature_engineering.py  # 特征工程工具
│   │
│   └── reporting/                # 报告生成组件
│       ├── __init__.py
│       ├── report_generator.py    # 报告生成器
│       └── visualizations.py      # 可视化工具
│
├── tests/                        # 测试代码
│   ├── __init__.py
│   ├── test_preprocessors/
│   │   ├── __init__.py
│   │   └── test_preprocessing_coordinator.py
│   ├── test_utils/
│   │   ├── __init__.py
│   │   └── test_timestamp_processor.py
│   └── test_models/
│       ├── __init__.py
│       └── test_multidimensional_detector.py
│
└── output/                       # 输出目录
    ├── reports/                  # 生成的报告
    └── visualizations/           # 生成的可视化，这是我之前的目录， """

```