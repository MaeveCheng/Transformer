#!/usr/bin/env python
import requests
import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
import logging
import json
import signal
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置
DATA_DIR = "/workspace/Maeve/crypto"
BINANCE_S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_DOWNLOAD_BASE = "https://data.binance.vision"
PROGRESS_FILE = "/workspace/Maeve/download_progress.json"

# 全局变量用于优雅退出
shutdown = False

def signal_handler(signum, frame):
    """处理中断信号"""
    global shutdown
    logger.info("\n\n⚠️  Received interrupt signal. Saving progress and shutting down...")
    shutdown = True

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_progress():
    """加载进度文件"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "completed_symbols": [],
        "failed_symbols": [],
        "no_data_symbols": [],
        "last_update": None
    }

def save_progress(progress):
    """保存进度"""
    progress["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_symbol_files(symbol: str) -> list:
    """获取某个symbol的所有可用文件"""
    import xml.etree.ElementTree as ET
    
    prefix = f"data/spot/daily/klines/{symbol}/1m/"
    files = []
    
    try:
        # 使用S3 API列出所有文件
        marker = None
        while True:
            params = {
                'prefix': prefix,
                'max-keys': 1000
            }
            if marker:
                params['marker'] = marker
                
            response = requests.get(BINANCE_S3_BASE, params=params, timeout=30)
            if response.status_code != 200:
                break
                
            root = ET.fromstring(response.content)
            
            # 查找所有文件
            for content in root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}Contents'):
                key = content.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key')
                if key is not None and key.text.endswith('.zip'):
                    # 提取文件名
                    filename = key.text.split('/')[-1]
                    files.append(filename)
            
            # 检查是否还有更多
            is_truncated = root.find('.//{http://s3.amazonaws.com/doc/2006-03-01/}IsTruncated')
            if is_truncated is not None and is_truncated.text.lower() == 'true':
                # 获取最后一个key作为marker
                contents = root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}Contents')
                if contents:
                    last_key = contents[-1].find('{http://s3.amazonaws.com/doc/2006-03-01/}Key')
                    if last_key is not None:
                        marker = last_key.text
                    else:
                        break
                else:
                    break
            else:
                break
                
    except Exception as e:
        logger.error(f"Error listing files for {symbol}: {e}")
    
    return sorted(files)

def download_file(symbol: str, filename: str, symbol_dir: str) -> bool:
    """下载单个文件"""
    global shutdown
    if shutdown:
        return False
        
    filepath = os.path.join(symbol_dir, filename)
    
    # 如果文件已存在且大小合理，跳过
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        if file_size > 1000:  # 至少1KB
            return True
    
    # 下载文件
    url = f"{BINANCE_DOWNLOAD_BASE}/data/spot/daily/klines/{symbol}/1m/{filename}"
    
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            # 先写入临时文件
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, 'wb') as f:
                f.write(response.content)
            # 原子性重命名
            os.rename(temp_filepath, filepath)
            return True
        return False
    except Exception as e:
        logger.debug(f"Failed to download {filename}: {e}")
        # 清理临时文件
        temp_filepath = filepath + ".tmp"
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return False

def download_symbol_all_files(args):
    """下载某个symbol的所有文件"""
    symbol, progress_dict = args
    global shutdown
    
    # 检查是否需要停止
    if shutdown:
        return f"INTERRUPTED: {symbol}"
    
    # 检查是否已经完成
    if symbol in progress_dict["completed_symbols"]:
        logger.info(f"  {symbol}: Already completed, skipping")
        return f"SKIPPED: {symbol}"
    
    try:
        # 创建symbol目录
        symbol_dir = os.path.join(DATA_DIR, f"crypto_{symbol}")
        ensure_dir(symbol_dir)
        
        logger.info(f"Processing {symbol}...")
        
        # 获取所有可用文件列表
        files = get_symbol_files(symbol)
        
        if not files:
            logger.warning(f"  {symbol}: No files found")
            return f"NO_DATA: {symbol}"
        
        # 统计已存在的文件
        existing_files = 0
        for filename in files:
            filepath = os.path.join(symbol_dir, filename)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                existing_files += 1
        
        logger.info(f"  {symbol}: {existing_files}/{len(files)} files already exist")
        
        # 下载所有文件
        downloaded = 0
        skipped = existing_files
        failed = 0
        
        for i, filename in enumerate(files):
            # 检查是否需要停止
            if shutdown:
                return f"INTERRUPTED: {symbol}"
                
            filepath = os.path.join(symbol_dir, filename)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                # 文件已存在，跳过
                continue
                
            if download_file(symbol, filename, symbol_dir):
                downloaded += 1
            else:
                failed += 1
            
            # 每100个文件显示进度
            if downloaded > 0 and downloaded % 100 == 0:
                logger.info(f"    {symbol}: Downloaded {downloaded} new files")
            
            # 每50个文件休息一下
            if downloaded > 0 and downloaded % 50 == 0:
                time.sleep(0.1)
        
        # 计算目录总大小
        total_size = 0
        for f in os.listdir(symbol_dir):
            filepath = os.path.join(symbol_dir, f)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
        
        total_size_mb = total_size / (1024 * 1024)
        
        logger.info(f"✅ {symbol}: Downloaded {downloaded} new files, skipped {skipped}, failed {failed}")
        logger.info(f"   Total size: {total_size_mb:.2f} MB")
        
        return f"SUCCESS: {symbol} - {downloaded} new files ({total_size_mb:.2f} MB)"
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return f"ERROR: {symbol} - {str(e)}"

def main():
    """主函数"""
    global shutdown
    ensure_dir(DATA_DIR)
    
    # 加载进度
    progress = load_progress()
    
    # 加载符号列表
    logger.info("Loading symbols...")
    with open('binance_vision_all_symbols.json', 'r') as f:
        symbols_data = json.load(f)
    
    # 准备所有符号
    all_symbols = []
    all_symbols.extend(symbols_data['usdt_pairs'])
    all_symbols.extend(symbols_data['btc_pairs'])
    all_symbols.extend(symbols_data['eth_pairs'])
    
    # 过滤已完成的符号
    remaining_symbols = [s for s in all_symbols if s not in progress["completed_symbols"]]
    
    total_symbols = len(all_symbols)
    completed_count = len(progress["completed_symbols"])
    remaining_count = len(remaining_symbols)
    
    logger.info(f"Total symbols: {total_symbols}")
    logger.info(f"  Already completed: {completed_count}")
    logger.info(f"  Remaining: {remaining_count}")
    logger.info(f"  USDT: {len(symbols_data['usdt_pairs'])}")
    logger.info(f"  BTC: {len(symbols_data['btc_pairs'])}")
    logger.info(f"  ETH: {len(symbols_data['eth_pairs'])}")
    
    if remaining_count == 0:
        logger.info("All symbols have been downloaded!")
        return
    
    # 配置进程数
    num_processes = 30
    logger.info(f"Using {num_processes} processes")
    
    logger.info("\n" + "="*60)
    if completed_count > 0:
        logger.info("RESUMING RAW data download")
    else:
        logger.info("Starting RAW data download")
    logger.info("Will download ALL zip files for each symbol")
    logger.info("Press Ctrl+C to save progress and exit")
    logger.info("="*60 + "\n")
    
    start_time = time.time()
    
    # 准备参数
    args_list = [(symbol, progress) for symbol in remaining_symbols]
    
    # 多进程下载
    results = []
    try:
        with Pool(processes=num_processes) as pool:
            # 使用imap_unordered以便更好地处理中断
            for result in pool.imap_unordered(download_symbol_all_files, args_list):
                results.append(result)
                
                # 更新进度
                if result.startswith("SUCCESS"):
                    symbol = result.split(":")[1].strip().split(" ")[0]
                    progress["completed_symbols"].append(symbol)
                elif result.startswith("NO_DATA"):
                    symbol = result.split(":")[1].strip()
                    progress["no_data_symbols"].append(symbol)
                elif result.startswith("ERROR"):
                    symbol = result.split(":")[1].strip().split(" ")[0]
                    progress["failed_symbols"].append(symbol)
                
                # 每10个符号保存一次进度
                if len(results) % 10 == 0:
                    save_progress(progress)
                    logger.info(f"Progress saved: {len(progress['completed_symbols'])}/{total_symbols} completed")
                
                # 检查是否需要停止
                if shutdown:
                    logger.info("Stopping download...")
                    pool.terminate()
                    break
                    
    except KeyboardInterrupt:
        logger.info("\nReceived interrupt, saving progress...")
    
    # 保存最终进度
    save_progress(progress)
    
    # 统计
    session_success = sum(1 for r in results if r.startswith("SUCCESS"))
    session_no_data = sum(1 for r in results if r.startswith("NO_DATA"))
    session_error = sum(1 for r in results if r.startswith("ERROR"))
    session_interrupted = sum(1 for r in results if r.startswith("INTERRUPTED"))
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*60)
    if shutdown or session_interrupted > 0:
        logger.info("DOWNLOAD PAUSED - Progress saved!")
    else:
        logger.info("DOWNLOAD COMPLETE!")
    logger.info("="*60)
    logger.info(f"Session time: {elapsed_time/3600:.2f} hours")
    logger.info(f"Session success: {session_success}")
    logger.info(f"Session no data: {session_no_data}")
    logger.info(f"Session errors: {session_error}")
    if session_interrupted > 0:
        logger.info(f"Session interrupted: {session_interrupted}")
    logger.info(f"\nTotal completed: {len(progress['completed_symbols'])}/{total_symbols}")
    logger.info(f"Total remaining: {total_symbols - len(progress['completed_symbols'])}")
    
    # 保存会话报告
    session_report = {
        "session_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_hours": elapsed_time/3600,
        "session_success": session_success,
        "session_no_data": session_no_data,
        "session_errors": session_error,
        "session_interrupted": session_interrupted,
        "total_completed": len(progress["completed_symbols"]),
        "total_remaining": total_symbols - len(progress["completed_symbols"]),
        "total_symbols": total_symbols
    }
    
    report_file = os.path.join(DATA_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w') as f:
        json.dump(session_report, f, indent=2)
    
    logger.info(f"\nSession report: {report_file}")
    logger.info(f"Progress file: {PROGRESS_FILE}")
    
    if shutdown or session_interrupted > 0:
        logger.info("\n💡 To resume download, run the script again")
        logger.info("   It will automatically continue from where it left off")

if __name__ == "__main__":
    main()