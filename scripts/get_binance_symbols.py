#!/usr/bin/env python
"""
获取Binance Data Vision上所有可用的交易对
"""
import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_symbols():
    """从Binance S3获取所有可用的symbol"""
    base_url = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    prefix = "data/spot/daily/klines/"
    
    symbols = set()
    marker = None
    
    logger.info("Fetching all available symbols from Binance Data Vision...")
    
    while True:
        # 构建请求参数
        params = {
            'prefix': prefix,
            'delimiter': '/',
            'max-keys': 1000
        }
        if marker:
            params['marker'] = marker
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to fetch data: {response.status_code}")
                break
            
            # 解析XML响应
            root = ET.fromstring(response.content)
            
            # 获取所有CommonPrefixes（这些是symbol目录）
            for prefix_elem in root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}CommonPrefixes'):
                prefix_text = prefix_elem.find('{http://s3.amazonaws.com/doc/2006-03-01/}Prefix')
                if prefix_text is not None:
                    # 从路径中提取symbol名称
                    # 格式: data/spot/daily/klines/BTCUSDT/
                    parts = prefix_text.text.rstrip('/').split('/')
                    if len(parts) >= 5:
                        symbol = parts[4]
                        symbols.add(symbol)
            
            # 检查是否还有更多数据
            is_truncated = root.find('.//{http://s3.amazonaws.com/doc/2006-03-01/}IsTruncated')
            if is_truncated is not None and is_truncated.text.lower() == 'true':
                # 获取NextMarker
                next_marker = root.find('.//{http://s3.amazonaws.com/doc/2006-03-01/}NextMarker')
                if next_marker is not None:
                    marker = next_marker.text
                else:
                    # 如果没有NextMarker，使用最后一个Prefix作为marker
                    prefixes = root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}CommonPrefixes')
                    if prefixes:
                        last_prefix = prefixes[-1].find('{http://s3.amazonaws.com/doc/2006-03-01/}Prefix')
                        if last_prefix is not None:
                            marker = last_prefix.text
                        else:
                            break
                    else:
                        break
            else:
                break
                
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            break
    
    return sorted(list(symbols))

def categorize_symbols(symbols):
    """将symbols按交易对类型分类"""
    usdt_pairs = []
    btc_pairs = []
    eth_pairs = []
    
    for symbol in symbols:
        if symbol.endswith('USDT'):
            usdt_pairs.append(symbol)
        elif symbol.endswith('BTC'):
            btc_pairs.append(symbol)
        elif symbol.endswith('ETH'):
            eth_pairs.append(symbol)
    
    return {
        'usdt_pairs': sorted(usdt_pairs),
        'btc_pairs': sorted(btc_pairs),
        'eth_pairs': sorted(eth_pairs)
    }

def main():
    # 获取所有symbols
    symbols = get_all_symbols()
    
    if not symbols:
        logger.error("No symbols found!")
        return
    
    logger.info(f"Found {len(symbols)} symbols")
    
    # 分类symbols
    categorized = categorize_symbols(symbols)
    
    # 显示统计
    logger.info("\nSymbol Statistics:")
    logger.info(f"  USDT pairs: {len(categorized['usdt_pairs'])}")
    logger.info(f"  BTC pairs: {len(categorized['btc_pairs'])}")
    logger.info(f"  ETH pairs: {len(categorized['eth_pairs'])}")
    logger.info(f"  BNB pairs: {len(categorized['bnb_pairs'])}")
    logger.info(f"  Other pairs: {len(categorized['other_pairs'])}")
    logger.info(f"  Total: {categorized['total_symbols']}")
    
    # 保存到文件
    output_file = 'binance_vision_all_symbols.json'
    with open(output_file, 'w') as f:
        json.dump(categorized, f, indent=2)
    
    logger.info(f"\n✅ Symbols saved to {output_file}")


if __name__ == "__main__":
    main()