#!/usr/bin/env python3 -u
"""
日志监控脚本
监控训练日志文件，检测异常并自动恢复训练
"""

import os
import sys
import time
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import signal

# 强制关闭输出缓冲
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# ==================== 配置部分 ====================
# 监控配置
CHECK_INTERVAL = 30  # 检查间隔（秒）
LINES_TO_READ = 200  # 每次读取的行数
RECOVERY_SLEEP = 300  # 恢复后休眠时间（5分钟）

# 用户可自定义的关键字数组
ERROR_KEYWORDS = [
    "NaN or Inf",  # 首要监控的关键字
    "Gradient explosion detected at step",
    "Failed to process chunk"
    # 可以添加更多自定义关键字，例如：
    # "CUDA out of memory",
    # "RuntimeError",
    # "Segmentation fault",
]

# 路径配置
LOG_DIR = "/workspace2/Louie/Transformer2/Transformer"
#LOG_DIR = "/workspace2/Haiyang/Transformer"
LOG_FILES = ["0.log", "1.log", "2.log", "3.log"]  # 可扩展
#LOG_FILES = ["0.log", "1.log"]  # 可扩展
CHECKPOINT_DIR = "/workspace2/Louie/Transformer2/Transformer/logs/default/checkpoints"
#CHECKPOINT_DIR = "/workspace2/Haiyang/Transformer/logs/default/checkpoints"
MULTI_NODE_SCRIPT = "multi_node.sh"  # 相对于当前脚本的路径

# Telegram配置（可选）
TELEGRAM_ENABLED = True  # 设置为True启用Telegram通知
TELEGRAM_BOT_TOKEN = "6892052753:AAGKvsyN21jTwb5ChCFus9B07k7hZ1R1pY0"
TELEGRAM_CHAT_ID = "-1002097883878"
PROXY_URL = "http://127.0.0.1:8118"  # 留空表示不使用代理，或设置为 "http://proxy:port"

# ==================== 日志监控器类 ====================
class LogMonitor:
    def __init__(self):
        self.log_dir = Path(LOG_DIR)
        self.log_files = LOG_FILES
        self.file_positions = {}  # 记录每个文件的读取位置
        self.running = True
        self.telegram_notifier = None  # 将在需要时初始化
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """处理退出信号"""
        print(f"\n[{self.get_timestamp()}] 收到退出信号，正在停止监控...")
        self.running = False
        sys.exit(0)
    
    def get_timestamp(self):
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def read_last_lines(self, file_path: Path, n: int = 200) -> List[str]:
        """读取文件的最后n行"""
        try:
            # 使用tail命令读取最后n行，类似ddp_monitor.py的方式
            result = subprocess.run(
                ['tail', '-n', str(n), str(file_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                return result.stdout.splitlines()
            else:
                return []
                
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"[{self.get_timestamp()}] 读取文件 {file_path} 出错: {e}")
            return []
    
    def check_for_errors(self, lines: List[str], log_file: str) -> Optional[Tuple[str, str, int]]:
        """检查日志行中是否包含错误关键字"""
        # 只检查最新的50行，类似ddp_monitor.py
        recent_lines = lines[-100:] if len(lines) > 100 else lines
        
        for i, line in enumerate(recent_lines):
            for keyword in ERROR_KEYWORDS:
                if keyword in line:
                    # 返回：(匹配的关键字, 包含关键字的行, 在recent_lines中的行号)
                    return (keyword, line.strip(), i)
        return None
    
    def extract_training_progress(self, lines: List[str]) -> Optional[dict]:
        """从日志中提取训练进度"""
        import re
        
        # 从后往前查找，找到最新的进度信息
        for line in reversed(lines):
            # 匹配训练进度的模式 (适配多种格式)
            # 格式1: Epoch 0, Step 12004, Accuracy: 0.701
            # 格式2: Epoch 0: | 5/? [00:36<00:00, train_accuracy_step=0.701
            # 格式3: Step 12004 - Weight changes: {...}, Total: 0.000043
            # 格式4: Epoch 0: |          | 1934/? [08:07<00:00
            patterns = [
                r'Epoch\s+(\d+).*Step\s+(\d+).*Accuracy:\s+([\d.]+)',
                r'Epoch\s+(\d+):\s*\|[^|]*\|\s*(\d+)/\?.*train_accuracy_step=([\d.]+)',  # 匹配带步数的进度条
                r'Epoch\s+(\d+):.*train_accuracy_step=([\d.]+)',
                r'Step\s+(\d+)\s+-.*train_accuracy_step=([\d.]+)',
                r'Step\s+(\d+)\s+-.*Accuracy.*?([0-9.]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) == 3:
                        return {
                            'epoch': int(match.group(1)),
                            'step': int(match.group(2)),
                            'accuracy': float(match.group(3))
                        }
                    elif len(match.groups()) == 2:
                        # 对于只有epoch和accuracy的情况
                        return {
                            'epoch': int(match.group(1)),
                            'step': 0,
                            'accuracy': float(match.group(2))
                        }
            
            # 单独处理没有accuracy的进度条格式
            # 格式: Epoch 0: |          | 1934/? [08:07<00:00
            progress_pattern = r'Epoch\s+(\d+):\s*\|[^|]*\|\s*(\d+)/\?'
            match = re.search(progress_pattern, line)
            if match:
                return {
                    'epoch': int(match.group(1)),
                    'step': int(match.group(2)),
                    'accuracy': 0.0  # 没有accuracy信息时返回0
                }
                
        return None
    
    def check_log_update_time(self, log_path: Path) -> tuple[bool, float]:
        """检查日志文件更新时间，返回(是否超时, 分钟数)"""
        try:
            if log_path.exists():
                last_modified = log_path.stat().st_mtime
                minutes_since_update = (time.time() - last_modified) / 60
                return minutes_since_update > 5, minutes_since_update
            return True, float('inf')
        except Exception:
            return True, float('inf')
    
    def monitor_logs(self) -> Optional[Tuple[str, str, str, datetime, dict]]:
        """监控所有日志文件，返回第一个发现的错误和训练进度"""
        
        stale_logs = []  # 记录超时的日志文件
        
        for log_file in self.log_files:
            log_path = self.log_dir / log_file
            
            # 检查日志更新时间
            is_stale, minutes_since = self.check_log_update_time(log_path)
            
            # 打印文件最后更新时间
            if log_path.exists():
                try:
                    last_modified = datetime.fromtimestamp(log_path.stat().st_mtime)
                    print(f"[{self.get_timestamp()}] {log_file} - 最后更新: {last_modified.strftime('%Y-%m-%d %H:%M:%S')} ({minutes_since:.1f}分钟前)")
                except Exception:
                    print(f"[{self.get_timestamp()}] {log_file} - 无法获取更新时间")
            else:
                print(f"[{self.get_timestamp()}] {log_file} - 文件不存在")
            
            # 如果日志超时，添加到列表
            if is_stale:
                stale_logs.append((log_file, minutes_since))
            
            # 读取最后200行
            lines = self.read_last_lines(log_path, LINES_TO_READ)
            
            if not lines:
                continue
            
            # 提取训练进度
            progress = self.extract_training_progress(lines)
            if progress:
                print(f"[{self.get_timestamp()}] {log_file} - 训练进度: Epoch {progress['epoch']}, Step {progress['step']}, Accuracy {progress['accuracy']:.4f}")
            
            # 检查错误
            error_info = self.check_for_errors(lines, log_file)
            
            if error_info:
                keyword, error_line, line_idx = error_info
                error_time = datetime.now()
                
                print(f"[{self.get_timestamp()}] 发现异常！")
                print(f"  文件: {log_file}")
                print(f"  关键字: \"{keyword}\"")
                print(f"  错误行: {error_line}")
                print(f"  异常时间: {error_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                return (log_file, keyword, error_line, error_time, progress)
        
        # 如果有任意日志超过3分钟没更新，发送通知
        if stale_logs:
            self.send_log_stale_notification(stale_logs)
        
        print(f"[{self.get_timestamp()}] 检查完成，未发现异常")
        return None
    
    def send_log_stale_notification(self, stale_logs):
        """发送日志长时间未更新的通知"""
        # 打印所有超时的日志文件信息
        for log_file, minutes_since in stale_logs:
            print(f"[{self.get_timestamp()}] 警告：日志文件 {log_file} 超过3分钟无更新 ({minutes_since:.1f}分钟)")
        
        # 发送Telegram通知（受5分钟间隔限制）
        if not self.telegram_notifier:
            self.telegram_notifier = TelegramNotifier()
        
        # 构建包含所有超时日志的消息
        stale_files_info = "\n".join([f"• {log_file}: {minutes_since:.1f}分钟无更新" for log_file, minutes_since in stale_logs])
        
        message = f"""🚨 *训练异常检测*

*详情:* 以下日志文件超过3分钟无更新：
{stale_files_info}

*建议:* 请检查对应的训练进程是否正常运行"""
        
        self.telegram_notifier.send_message(message, level='warning')


# ==================== Checkpoint管理器类 ====================
class CheckpointManager:
    def __init__(self):
        self.checkpoint_dir = Path(CHECKPOINT_DIR)
    
    def get_timestamp(self):
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def find_best_checkpoint(self, error_time: datetime) -> Optional[Path]:
        """根据错误时间找到最合适的checkpoint"""
        print(f"[{self.get_timestamp()}] 扫描checkpoint目录...")
        
        try:
            # 获取所有.ckpt文件（排除last.ckpt）
            ckpt_files = [
                f for f in self.checkpoint_dir.glob("*.ckpt")
                if f.name != "last.ckpt"
            ]
            
            if not ckpt_files:
                print(f"[{self.get_timestamp()}] 错误：未找到任何checkpoint文件")
                return None
            
            # 获取文件创建时间并排序
            ckpt_with_time = []
            for ckpt in ckpt_files:
                create_time = datetime.fromtimestamp(ckpt.stat().st_mtime)
                ckpt_with_time.append((ckpt, create_time))
            
            # 按时间排序（从新到旧）
            ckpt_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # 找到错误时间之前最新的checkpoint
            for ckpt, create_time in ckpt_with_time:
                if create_time < error_time:
                    print(f"[{self.get_timestamp()}] 选择checkpoint: {ckpt.name}")
                    print(f"  创建时间: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    return ckpt
            
            # 如果所有checkpoint都比错误时间新，选择最旧的
            oldest_ckpt = ckpt_with_time[-1][0]
            print(f"[{self.get_timestamp()}] 所有checkpoint都比错误时间新，选择最旧的: {oldest_ckpt.name}")
            return oldest_ckpt
            
        except Exception as e:
            print(f"[{self.get_timestamp()}] 扫描checkpoint目录出错: {e}")
            return None
    
    def rename_checkpoint(self, checkpoint: Path) -> bool:
        """将选中的checkpoint重命名为last.ckpt"""
        try:
            last_ckpt = self.checkpoint_dir / "last.ckpt"
            
            # 如果last.ckpt已存在，先删除
            if last_ckpt.exists():
                last_ckpt.unlink()
                print(f"[{self.get_timestamp()}] 已删除原有的last.ckpt")
            
            # 复制而不是移动，保留原始checkpoint
            import shutil
            shutil.copy2(checkpoint, last_ckpt)
            print(f"[{self.get_timestamp()}] 重命名为last.ckpt成功")
            return True
            
        except Exception as e:
            print(f"[{self.get_timestamp()}] 重命名checkpoint失败: {e}")
            return False


# ==================== Telegram通知类 ====================
class TelegramNotifier:
    def __init__(self):
        self.enabled = TELEGRAM_ENABLED
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.proxy_url = PROXY_URL
        self.min_interval = 300  # 最小通知间隔：5分钟（300秒）
        self.last_notification_time = 0  # 记录最后一次通知时间（不区分类型）
    
    def get_timestamp(self):
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def send_message(self, message: str, level: str = 'info') -> bool:
        """发送Telegram消息"""
        if not self.enabled:
            return True
        
        if self.bot_token == "YOUR_BOT_TOKEN" or self.chat_id == "YOUR_CHAT_ID":
            print(f"[{self.get_timestamp()}] 警告: Telegram未配置，请设置bot_token和chat_id")
            return False
        
        # 检查是否距离上次通知已经超过5分钟
        current_time = time.time()
        time_since_last = current_time - self.last_notification_time
        
        if time_since_last < self.min_interval:
            remaining_time = self.min_interval - time_since_last
            print(f"[{self.get_timestamp()}] Telegram通知被限制：距离上次通知还需等待 {remaining_time:.0f} 秒")
            return False
        
        try:
            import requests
            
            print(f"[{self.get_timestamp()}] 发送Telegram通知...")
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": "10ms模型 \n\n" + message,
                "parse_mode": "Markdown",
                "disable_notification": level == 'info'  # info级别静默通知
            }
            
            # 设置代理
            proxies = {}
            if self.proxy_url:
                proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url
                }
            
            response = requests.post(url, json=data, proxies=proxies, timeout=10)
            
            if response.status_code == 200:
                print(f"[{self.get_timestamp()}] Telegram通知发送成功")
                # 更新最后通知时间
                self.last_notification_time = current_time
                return True
            else:
                print(f"[{self.get_timestamp()}] Telegram通知发送失败: {response.text}")
                return False
                
        except ImportError:
            print(f"[{self.get_timestamp()}] 警告：未安装requests库，无法发送Telegram通知")
            return False
        except Exception as e:
            print(f"[{self.get_timestamp()}] Telegram通知发送失败: {e}")
            return False


# ==================== 训练恢复类 ====================
class TrainingRecovery:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.multi_node_script = self.script_dir / MULTI_NODE_SCRIPT
    
    def get_timestamp(self):
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def restart_training(self) -> bool:
        """调用multi_node.sh重启训练"""
        try:
            print(f"[{self.get_timestamp()}] 调用{MULTI_NODE_SCRIPT}恢复训练...")
            
            # 确保脚本存在且可执行
            if not self.multi_node_script.exists():
                print(f"[{self.get_timestamp()}] 错误：{MULTI_NODE_SCRIPT}不存在")
                return False
            
            # 执行脚本
            result = subprocess.run(
                ["bash", str(self.multi_node_script)],
                cwd=str(self.script_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"[{self.get_timestamp()}] 训练恢复成功")
                return True
            else:
                print(f"[{self.get_timestamp()}] 训练恢复失败")
                print(f"  错误输出: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[{self.get_timestamp()}] 调用{MULTI_NODE_SCRIPT}失败: {e}")
            return False


# ==================== 主函数 ====================
def main():
    """主监控循环"""
    print("="*60)
    print("日志监控脚本启动")
    print(f"监控目录: {LOG_DIR}")
    print(f"监控文件: {', '.join(LOG_FILES)}")
    print(f"检查间隔: {CHECK_INTERVAL}秒")
    print(f"读取行数: {LINES_TO_READ}")
    print(f"监控关键字: {', '.join(ERROR_KEYWORDS)}")
    print("="*60)
    
    # 初始化各个组件
    monitor = LogMonitor()
    checkpoint_manager = CheckpointManager()
    notifier = TelegramNotifier()
    recovery = TrainingRecovery()
    
    # 启动延迟
    startup_delay = 180  # 3分钟 = 180秒
    print(f"[{monitor.get_timestamp()}] 等待 {startup_delay} 秒（3分钟）后开始监控...")
    print(f"[{monitor.get_timestamp()}] 这样可以让训练进程充分启动...")
    
    # 显示倒计时（每30秒更新一次）
    for remaining in range(startup_delay, 0, -30):
        print(f"[{monitor.get_timestamp()}] 剩余等待时间: {remaining} 秒...")
        time.sleep(min(30, remaining))
    
    print(f"[{monitor.get_timestamp()}] 开始监控日志文件...")
    
    # 主循环
    while monitor.running:
        try:
            # 监控日志
            error_info = monitor.monitor_logs()
            
            if error_info:
                log_file, keyword, error_line, error_time, progress = error_info

                message = f"""🚨 *训练异常检测*

*节点:* {log_file}
*关键字:* {keyword}"""
                # 添加训练进度信息（如果有）
                if progress:
                    message += f"\n*训练进度:* Epoch {progress['epoch']}, Step {progress['step']}, Accuracy {progress['accuracy']:.4f}"
                    # 转义error_line中的特殊字符
                    escaped_error_line = error_line[:200].replace('`', '\\`').replace('*', '\\*').replace('_', '\\_').replace('[', '\\[').replace(']', '\\]')
                    message += f"\n*错误行:* `{escaped_error_line}`"
                    notifier.send_message(message, level='critical')
                
#                 # 1. 选择合适的checkpoint
#                 best_checkpoint = checkpoint_manager.find_best_checkpoint(error_time)
#                 if best_checkpoint:
#                     # 2. 重命名checkpoint
#                     if checkpoint_manager.rename_checkpoint(best_checkpoint):
                        
#                         # 3. 发送Telegram通知（仅发送异常信息）
#                         message = f"""🚨 *训练异常检测*

# *节点:* {log_file}
# *关键字:* {keyword}"""
                        
#                         # 添加训练进度信息（如果有）
#                         if progress:
#                             message += f"\n*训练进度:* Epoch {progress['epoch']}, Step {progress['step']}, Accuracy {progress['accuracy']:.4f}"
                        
#                         # 转义error_line中的特殊字符
#                         escaped_error_line = error_line[:200].replace('`', '\\`').replace('*', '\\*').replace('_', '\\_').replace('[', '\\[').replace(']', '\\]')
                        
#                         message += f"\n*错误行:* `{escaped_error_line}`"
#                         message += f"\n*选择Checkpoint:* {best_checkpoint.name}"
                        
#                         notifier.send_message(message, level='critical')
                        
#                         # 4. 重启训练（只打印日志，不发送Telegram）
#                         if recovery.restart_training():
#                             # 5. 成功后休眠5分钟
#                             print(f"[{monitor.get_timestamp()}] 训练恢复成功，休眠5分钟...")
#                             time.sleep(RECOVERY_SLEEP)
#                             print(f"[{monitor.get_timestamp()}] 休眠结束，继续监控...")
#                         else:
#                             print(f"[{monitor.get_timestamp()}] 训练恢复失败，继续监控...")
                else:
                    print(f"[{monitor.get_timestamp()}] 未找到合适的checkpoint，跳过恢复")
            
            # 等待下一次检查
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{monitor.get_timestamp()}] 监控过程出错: {e}")
            time.sleep(CHECK_INTERVAL)
    
    print("\n监控脚本已停止")


if __name__ == "__main__":
    main()
