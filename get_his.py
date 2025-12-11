import streamlit as st
import requests
import pandas as pd

def fetch_eps_data(stock_code):
    """
    尝试从网络获取EPS数据
    :param stock_code: 股票代码
    :return: 包含EPS数据的字典
    """
    try:
        url = f"https://finance.sina.com.cn/realstock/company/{stock_code}/nc.shtml"  # 示例：新浪财经个股页面
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试多种方式查找EPS数据
        # 方法1: 查找特定class
        eps_tag = soup.find('span', {'class': 'eps_value'})
        if eps_tag:
            return {'stock_code': stock_code, 'eps': eps_tag.text}
        
        # 方法2: 查找包含'EPS'文本的元素
        eps_elements = soup.find_all(text=lambda text: text and 'EPS' in text)
        for elem in eps_elements:
            parent = elem.parent
            # 尝试找到相邻的数值
            next_sibling = parent.next_sibling
            if next_sibling and next_sibling.string:
                # 提取数字部分
                import re
                numbers = re.findall(r'\d+\.?\d*', next_sibling.string)
                if numbers:
                    return {'stock_code': stock_code, 'eps': numbers[0]}
        
        # 如果都找不到，返回N/A
        return {'stock_code': stock_code, 'eps': 'N/A'}
    except Exception as e:
        # 出错时返回N/A，但不抛出异常以避免程序中断
        print(f"获取EPS数据时出错: {e}")
        return {'stock_code': stock_code, 'eps': 'N/A'}

def get_stock_history(stock_code, start_date, end_date):
    """
    抓取股票历史数据（支持沪深A股）
    :param stock_code: 股票代码（如 "sh600519" 贵州茅台）
    :param start_date: 开始日期 "20230101"
    :param end_date: 结束日期 "20231231"
    :return: DataFrame格式数据
    """
    url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {
    "symbol": stock_code,
    "scale": 240,  # 日线数据
    "ma": "no",
    "datalen": 1024  # 最大返回条数
    }

    try:
        headers = {
        "User - Agent": "Mozilla/5.0(Windows NT 10.0;Win64;x64) AppleWebKit/537.36"
        }
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()

        df = pd.DataFrame(data)
        df["day"] = pd.to_datetime(df["day"])
        df.rename(columns={
        "day": "日期",
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "volume": "成交量"
        }, inplace = True)

        # 数据类型转换
        df = df.astype({
        "开盘价": "float", "最高价": "float", "最低价": "float",
        "收盘价": "float", "成交量": "float"
        })

        # 日期范围过滤（新浪接口可能返回多余数据）
        mask = (df["日期"] >= pd.to_datetime(start_date)) & (df["日期"] <= pd.to_datetime(end_date))
        return df[mask].sort_values("日期")

    except Exception as e:
        st.error(f"历史数据抓取失败: {e}")
        st.stop()

def fetch_eps_data(stock_code):
    """
    尝试从网络获取EPS数据
    :param stock_code: 股票代码
    :return: 包含EPS数据的字典
    """
    try:
        url = f"https://finance.sina.com.cn/realstock/company/{stock_code}/nc.shtml"  # 示例：新浪财经个股页面
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试多种方式查找EPS数据
        # 方法1: 查找特定class
        eps_tag = soup.find('span', {'class': 'eps_value'})
        if eps_tag:
            return {'stock_code': stock_code, 'eps': eps_tag.text}
        
        # 方法2: 查找包含'EPS'文本的元素
        eps_elements = soup.find_all(text=lambda text: text and 'EPS' in text)
        for elem in eps_elements:
            parent = elem.parent
            # 尝试找到相邻的数值
            next_sibling = parent.next_sibling
            if next_sibling and next_sibling.string:
                # 提取数字部分
                import re
                numbers = re.findall(r'\d+\.?\d*', next_sibling.string)
                if numbers:
                    return {'stock_code': stock_code, 'eps': numbers[0]}
        
        # 如果都找不到，返回N/A
        return {'stock_code': stock_code, 'eps': 'N/A'}
    except Exception as e:
        # 出错时返回N/A，但不抛出异常以避免程序中断
        print(f"获取EPS数据时出错: {e}")
        return {'stock_code': stock_code, 'eps': 'N/A'}