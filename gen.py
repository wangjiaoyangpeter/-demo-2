import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from get_his import get_stock_history
from pathlib import Path

def file_road():
    BASE_DIR = Path().parent
    DATA_DIR = BASE_DIR / "data"
    CSV_PATH = DATA_DIR / "records.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return CSV_PATH

def gen_stock_history():
    stock_history = None
    price_change_percent = None  # 初始化变量
    with st.form("stock_history"):
        stock_code = st.text_input("请输入股票代码", placeholder="例如: sh600519")
        start_date = st.date_input("输入起始日期", datetime.today() - timedelta(days=120),max_value=datetime.today())
        end_date = st.date_input("输入终止日期", datetime.today(),min_value=start_date,max_value=datetime.today())
        submit_button= st.form_submit_button ("查询",use_container_width=True )
    
    CSV_PATH = file_road()
    if submit_button and stock_code:
        stock_history = get_stock_history(stock_code, start_date, end_date)
        st.success(f"成功获取 {stock_code} 的历史数据！共 {len(stock_history)} 条记录")
        stock_history.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    elif submit_button:
        st.warning("请输入股票代码")
    elif CSV_PATH.exists():
        stock_history = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    return stock_history,stock_code

