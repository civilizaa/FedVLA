"""FedVLA relay —— 部署入口。

sites 平台会注入 PORT 环境变量；本服务监听 0.0.0.0:$PORT。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import main  # noqa: E402

if __name__ == "__main__":
    main()
