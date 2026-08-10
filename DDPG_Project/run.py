"""
PER-DDPG 云-雾-边协同任务卸载系统
运行: python run.py
访问: http://127.0.0.1:5000
"""
import sys, os
os.environ['NO_PROXY'] = '*'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from web.app import create_app


def main():
    config = AppConfig()
    app, socketio = create_app(config)
    print(f"\n{'='*60}")
    print(f"  PER-DDPG 云-雾-边协同任务卸载系统")
    print(f"  论文: 优先经验回放DDPG + DAG依赖任务分解")
    print(f"  访问: http://{config.host}:{config.port}")
    print(f"  架构: 1云 + {config.env.fog_node_num}雾 + {config.env.edge_device_num}边缘设备")
    print(f"{'='*60}\n")
    socketio.run(app, host=config.host, port=config.port, debug=config.debug)


if __name__ == '__main__':
    main()
