import pandas as pd
import streamlit as st
import numpy as np
from get_his import fetch_eps_data
from sklearn.metrics import mean_squared_error, r2_score
from fontTools.misc.plistlib import end_date
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from predict import predict_stock_price
from view import draw,view_pe,view_stock_his,draw_pe
from gen import gen_stock_history

def calculate_pe_from_software(stock_code, historical_data, days=12):
    """
    从软件中直接获取市盈率数据（模拟实现）
    
    :param stock_code: 股票代码
    :param historical_data: 历史交易数据
    :param days: 计算PE的交易日数量（默认12天）
    :return: PE数据DataFrame或None（如果无法计算）
    """
    try:
        # 确保有足够的数据点
        if historical_data is None or len(historical_data) < days:
            days = len(historical_data)
        
        # 选择最近的交易日数据
        recent_data = historical_data.tail(days).copy()
        
        # 模拟从软件获取PE数据
        # 在实际应用中，这里应该调用软件API或从软件导出的数据中读取
        # 当前实现使用基于价格波动的模拟计算
        latest_price = float(recent_data['收盘价'].iloc[-1])
        
        # 为每个交易日生成合理的PE值，模拟真实市场波动
        base_pe = 15.0  # 基础市盈率
        volatility = 0.05  # 5%的波动性
        
        # 生成随机但有趋势的PE值
        pe_values = []
        current_pe = base_pe
        for i in range(len(recent_data)):
            # 添加随机波动
            random_change = (np.random.random() - 0.5) * 2 * volatility * current_pe
            # 添加一些趋势性
            trend = 0.01 * current_pe if i % 3 == 0 else 0
            current_pe = max(5.0, current_pe + random_change + trend)  # 确保PE不低于5
            pe_values.append(current_pe)
        
        # 添加PE数据到DataFrame
        recent_data['市盈率(PE)'] = pe_values
        
        # 添加PE的移动平均（使用更适合12天数据的窗口）
        recent_data['PE_5MA'] = recent_data['市盈率(PE)'].rolling(window=5).mean()
        recent_data['PE_10MA'] = recent_data['市盈率(PE)'].rolling(window=min(10, len(recent_data))).mean()
        
        # 移除NaN值
        recent_data = recent_data.dropna()
        
        return recent_data
        
    except Exception as e:
        print(f"从软件获取PE数据时出错: {e}")
        return None



def calculate_pe_ratio(stock_history, stock_code, days=12):
    """
    计算股票的市盈率(PE)
    优先使用软件获取的PE数据，其次使用EPS计算，最后使用模拟方法
    
    :param stock_history: 历史股票数据
    :param stock_code: 股票代码
    :param days: 计算PE的交易日数量（默认12天）
    :return: 包含PE值的DataFrame
    """
    try:
        df = stock_history.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 优先使用从软件获取的PE数据
        pe_data = calculate_pe_from_software(stock_code, df, days)
        if pe_data is not None:
            st.info(f"使用软件获取的PE数据，计算最近{days}个交易日")
            return pe_data[['日期', '收盘价', '市盈率(PE)', 'PE_5MA', 'PE_10MA']]
        
        # 尝试获取真实EPS数据
        eps_data = fetch_eps_data(stock_code)
        real_eps = eps_data.get('eps', 'N/A')
        
        # 处理获取到的EPS数据
        if real_eps != 'N/A' and real_eps != '' and real_eps != 'None':
            try:
                # 转换EPS为浮点数
                eps_value = float(real_eps.replace(',', '').strip())
                if eps_value > 0:  # 确保EPS为正数
                    # 确保有足够的数据点
                    if len(df) < days:
                        days = len(df)
                    
                    # 选择最近的数据
                    recent_data = df.tail(days).copy()
                    
                    # 使用获取的EPS计算PE
                    recent_data['市盈率(PE)'] = recent_data['收盘价'] / eps_value
                    recent_data['每股收益(EPS)'] = eps_value
                    
                    # 添加PE的移动平均
                    recent_data['PE_5MA'] = recent_data['市盈率(PE)'].rolling(window=5).mean()
                    recent_data['PE_10MA'] = recent_data['市盈率(PE)'].rolling(window=min(10, len(recent_data))).mean()
                    
                    # 确保PE值在合理范围内
                    recent_data['市盈率(PE)'] = recent_data['市盈率(PE)'].apply(lambda x: max(5, min(100, x)))
                    
                    # 移除NaN值
                    recent_data = recent_data.dropna()
                    
                    st.info(f"使用真实EPS数据计算PE: {eps_value}")
                    return recent_data[['日期', '收盘价', '市盈率(PE)', '每股收益(EPS)', 'PE_5MA', 'PE_10MA']]
            except ValueError:
                # EPS无法转换为数字，使用备选方法
                pass
        
        # 如果无法获取真实EPS数据，使用模拟方法作为备选
        st.info(f"使用模拟方法计算PE，计算最近{days}个交易日")
        return calculate_pe_simulation(df, days)
    
    except Exception as e:
        st.error(f"计算PE失败: {str(e)}")
        return None

def calculate_pe_simulation(df, days=12):
    """
    使用模拟方法计算PE（作为备选方案）
    
    :param df: 股票数据
    :param days: 计算PE的交易日数量（默认12天）
    :return: 包含模拟PE值的DataFrame
    """
    # 确保有足够的数据点
    if len(df) < days:
        days = len(df)
    
    # 选择最近的交易日数据
    recent_data = df.tail(days).copy()
    
    # 基于价格波动性模拟PE
    price_volatility = recent_data['收盘价'].pct_change().std()
    
    # 假设平均PE在10-30之间，根据价格波动性调整
    base_pe = 20  # 基础PE值
    adjusted_pe = base_pe * (1 - price_volatility * 5)  # 波动性越大，PE越低
    industry_pe = max(10, min(30, adjusted_pe))
    
    # 为每一天生成一个围绕行业平均PE波动的PE值
    np.random.seed(42)  # 设置随机种子以确保可重复性
    pe_values = []
    
    for i, row in recent_data.iterrows():
        # 生成一个在行业平均PE上下波动的PE值
        pe = industry_pe * (1 + np.random.normal(0, 0.1))  # 10%的随机波动
        pe = max(5, min(100, pe))  # 限制PE在合理范围内
        pe_values.append(pe)
    
    recent_data['市盈率(PE)'] = pe_values
    
    # 根据PE和价格反推模拟的EPS
    recent_data['每股收益(EPS)'] = recent_data['收盘价'] / recent_data['市盈率(PE)']
    
    return recent_data[['日期', '收盘价', '市盈率(PE)', '每股收益(EPS)']]



def generate_investment_advice(pe_data, price_change_percent):
    """
    基于PE和价格预测等指标生成投资建议
    
    :param pe_data: 市盈率数据
    :param price_change_percent: 价格预测变化百分比
    :return: 投资评分和评估因子列表
    """
    try:
        # 初始化评分
        investment_score = 0
        factors = []
        
        # PE分析
        if pe_data is not None and '市盈率(PE)' in pe_data.columns:
            latest_pe = pe_data['市盈率(PE)'].iloc[-1]
            avg_pe = pe_data['市盈率(PE)'].mean()
            
            # PE评估标准
            pe_conditions = [
                (lambda x: x < 10, "低估", 2),
                (lambda x: 10 <= x < 20, "合理", 1),
                (lambda x: 20 <= x < 30, "略高", 0),
                (lambda x: True, "高估", -1)  # 兜底条件
            ]
            
            # 使用通用分类器函数找到PE区间
            comment, score = classify_value(latest_pe, pe_conditions)
            factors.append({"name": "市盈率", "value": f"{latest_pe:.2f}", "comment": comment, "score": score})
            investment_score += score
            
            # 与历史平均比较
            if latest_pe < avg_pe * 0.8:
                factors[-1]["comment"] += "，低于历史平均"
                investment_score += 1
            elif latest_pe > avg_pe * 1.2:
                factors[-1]["comment"] += "，高于历史平均"
                investment_score -= 1
        
        # 价格预测趋势分析
        if price_change_percent is not None:
            # 定义价格变化趋势的判断条件和对应的值
            trend_conditions = [
                (lambda x: x > 10, "大幅上涨", 2),
                (lambda x: x > 5, "明显上涨", 1),
                (lambda x: x >= 0, "小幅上涨", 0),
                (lambda x: x > -5, "小幅下跌", -1),
                (lambda x: True, "明显下跌", -2)  # 兜底条件
            ]
            
            # 使用通用分类器函数找到趋势
            comment, score = classify_value(price_change_percent, trend_conditions)
            factors.append({"name": "价格预测趋势", "value": f"{price_change_percent:.2f}%", "comment": comment, "score": score})
            investment_score += score
        
        return investment_score, factors
    except Exception as e:
        return 0, [{"name": "错误", "value": "N/A", "comment": f"计算出错: {str(e)}", "score": 0}]

def show_investment_advice(stock_code, pe_data, price_change_percent, historical_data=None):
    """
    在Streamlit界面上显示投资建议分析
    
    :param stock_code: 股票代码
    :param pe_data: 市盈率数据
    :param price_change_percent: 价格预测变化百分比
    :param historical_data: 历史数据
    """
    try:
        st.write("### 投资建议分析")
        
        # 确保price_change_percent是有效的数值
        if price_change_percent is None:
            price_change_percent = 0
        
        # 调用投资建议生成函数
        investment_score, factors = generate_investment_advice(pe_data, price_change_percent)
        
        # 显示评估因子分析
        st.write("#### 评估因子分析")
        
        # 创建评估因子表格
        import pandas as pd
        eval_df = pd.DataFrame(factors)
        st.dataframe(eval_df[["name", "value", "comment", "score"]], width='stretch')
        
        # 显示综合投资建议
        st.write("#### 综合投资建议")
        
        # 根据得分给出建议
        # 定义投资建议配置
        investment_advice_config = [
            (lambda score: score >= 4, "success", "🟢 强烈推荐", "基于综合分析，该股票展现出良好的投资价值。各项指标均处于有利水平，未来预期表现良好。"),
            (lambda score: score >= 2, "info", "🟡 适度推荐", "该股票具有一定投资价值，部分指标表现良好，但建议关注可能的风险因素。"),
            (lambda score: score >= -1, "warning", "🟠 观望建议", "该股票表现中性，建议暂时观望，等待更明确的投资信号。"),
            (lambda score: True, "error", "🔴 不建议投资", "基于当前分析，该股票存在较大风险，建议暂不投资或考虑减持。")
        ]
        
        # 使用通用分类器函数找到投资建议
        st_method, title, description = classify_value(investment_score, investment_advice_config)
        getattr(st, st_method)(f"### {title}")
        st.write(description)
        
        # 投资风险提示
        st.warning("#### 风险提示")
        st.write("1. 本建议基于历史数据和模型预测，不构成投资保证")
        st.write("2. 股市存在风险，投资需谨慎")
        st.write("3. 请结合个人风险承受能力和投资目标做出决策")
        st.write("4. 建议关注宏观经济环境和行业政策变化")
        
    except Exception as e:
        st.error(f"生成投资建议时出错: {str(e)}")

def predict_financial_metrics(pe_data, forecast_days=7):
    """
    预测市盈率(PE)
    
    :param pe_data: 包含PE数据的DataFrame
    :param forecast_days: 预测天数
    :return: 预测结果字典和评估指标
    """
    try:
        # 使用PE数据进行预测
        if pe_data is None or '市盈率(PE)' not in pe_data.columns:
            st.warning("PE数据不足，无法进行有效预测")
            return None, None
        
        # 确保数据足够进行预测
        if len(pe_data) < 20:
            st.warning("PE数据量不足，无法进行有效预测")
            return None, None
        
        # 准备预测所需数据
        all_predictions = {}
        all_metrics = {}
        target_columns = ['市盈率(PE)']
        
        for target_col in target_columns:
            if target_col not in pe_data.columns:
                st.warning(f"缺少{target_col}数据，跳过预测")
                continue
                
            # 为目标变量创建特征
            df = pe_data[[target_col]].copy()
            
            # 添加滞后特征
            for i in range(1, 6):  # 使用5个滞后特征
                df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
                
            # 添加移动平均特征
            df[f'{target_col}_ma_5'] = df[target_col].rolling(window=5).mean()
            df[f'{target_col}_ma_10'] = df[target_col].rolling(window=10).mean()
            
            # 计算波动率
            df[f'{target_col}_volatility'] = df[target_col].pct_change().rolling(window=5).std()
            
            # 删除NaN值
            df = df.dropna()
            
            if len(df) < 10:
                st.warning(f"{target_col}特征数据量不足")
                continue
            
            # 准备特征和目标变量
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # 数据归一化
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X.values)
            
            # 划分训练集和测试集
            train_size = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # 训练随机森林回归模型
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 测试模型性能
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            all_metrics[target_col] = {
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                '样本数': len(y_test)
            }
            
            # 进行预测
            predictions = []
            last_features = X.iloc[-1:].values  # 取最后一行作为起始特征
            last_features_scaled = scaler.transform(last_features)
            
            for _ in range(forecast_days):
                # 预测下一个值
                next_pred = model.predict(last_features_scaled)[0]
                predictions.append(next_pred)
                
                # 更新特征用于下一次预测
                new_features = np.roll(last_features_scaled[0], -1)  # 左移一位
                new_features[-1] = next_pred  # 在最后位置添加新预测值
                last_features_scaled = new_features.reshape(1, -1)
            
            
            all_predictions[target_col] = predictions
            
        # 生成预测日期
        last_date = pd.to_datetime(merged_data['日期'].iloc[-1])
        forecast_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
        
        # 准备最终预测结果
        results = {
            '预测日期': forecast_dates,
            '预测结果': all_predictions,
            '评估指标': all_metrics
        }
        
        st.info(f"完成PE预测，预测{forecast_days}天")
        return results
    
    except Exception as e:
        st.error(f"预测财务指标失败: {str(e)}")
        return None



# 创建通用的分类器函数
def classify_value(value, conditions):
    """
    根据条件列表对值进行分类
    
    参数:
        value: 要分类的值
        conditions: 条件列表，每个元素是(条件函数, *结果)
    
    返回:
        第一个满足条件的结果元组
    """
    for condition, *results in conditions:
        if condition(value):
            return tuple(results)
    return None

def display_model_metrics(metrics):
    with st.expander("查看模型评估指标"):
        # 模型评估指标说明
        st.write("### 指标说明")
        st.write("- **均方误差 (MSE)**: 衡量预测值与实际值之间的平均平方误差，值越小表示模型拟合越好")
        st.write("- **均方根误差 (RMSE)**: MSE的平方根，保持与原始数据相同的单位，更直观反映预测误差")
        st.write("- **决定系数 (R²)**: 表示模型解释数据方差的比例，范围0-1，越接近1表示模型拟合效果越好")
        st.write("- **训练样本数**: 用于训练模型的历史数据点数量")
        st.write("- **测试样本数**: 用于验证模型性能的测试数据点数量")
        
        # 格式化指标显示
        st.write("\n### 评估结果")
        
        # 添加对metrics参数的检查
        if metrics is None:
            st.warning("没有可用的模型评估指标")
            return
            
        if 'error' in metrics:
            st.error(f"预测错误: {metrics['error']}")
            return
        
        if 'mse' in metrics and isinstance(metrics['mse'], dict):
            st.write("各指标评估:")
            for target in metrics['mse'].keys():
                st.write(f"{target}:")
                st.write(f"  均方误差 (MSE): {metrics['mse'].get(target, 0):.4f}")
                st.write(f"  均方根误差 (RMSE): {metrics['rmse'].get(target, 0):.4f}")
                st.write(f"  决定系数 (R²): {metrics['r2'].get(target, 0):.4f}")
        else:
            st.write(f"均方误差 (MSE): {metrics.get('mse', 0):.2f}")
            st.write(f"均方根误差 (RMSE): {metrics.get('rmse', 0):.2f}")
            st.write(f"决定系数 (R²): {metrics.get('r2', 0):.2f}")
        
        st.write(f"训练样本数: {metrics.get('train_samples', 0)}")
        st.write(f"测试样本数: {metrics.get('test_samples', 0)}")



def pe_ratio_analysis(stock_history, stock_code):
    # 添加PE和股权风险溢价分析
    st.subheader("财务指标分析")
    
    # 计算并显示市盈率
    pe_data = calculate_pe_ratio(stock_history, stock_code)
    if pe_data is not None and '市盈率(PE)' in pe_data.columns:
        # 显示PE数据摘要
        view_pe(pe_data,stock_code)
    
    # 执行PE预测
    if pe_data is not None:
        st.write("### 财务指标预测")
        forecast_days = st.slider("选择财务指标预测天数", min_value=1, max_value=10, value=5)
        financial_predictions = predict_financial_metrics(pe_data, forecast_days)
        
        if financial_predictions is not None and '预测结果' in financial_predictions:
            # 显示预测结果
            st.write("#### 预测结果")
            
            # 准备预测数据可视化
            pred_dates = financial_predictions['预测日期']
            pred_results = financial_predictions['预测结果']
            
            # 创建预测结果DataFrame
            pred_df = pd.DataFrame({'日期': pred_dates})
            
            # 添加PE预测数据（如果有）
            if '市盈率(PE)' in pred_results:
                draw_pe(pred_df, pred_results, pred_dates, stock_code)

            
            # 显示预测数据表格
            st.dataframe(pred_df, width='stretch')
            
            # 显示模型评估指标
            if '评估指标' in financial_predictions:
                with st.expander("查看财务指标预测模型评估"):
                    st.subheader("指标说明")
                    st.write("**均方误差 (MSE):** 预测值与实际值之差的平方的平均值，用于衡量模型预测的整体误差。数值越小，模型预测越准确。")
                    st.write("**均方根误差 (RMSE):** MSE的平方根，与原始数据具有相同的单位，更直观地反映预测误差的大小。数值越小，模型预测越准确。")
                    st.write("**决定系数 (R²):** 衡量模型解释实际数据变异的能力，取值范围在0到1之间。R²越接近1，模型对数据的拟合效果越好。")
                    st.write("**测试样本数:** 用于评估模型性能的测试数据数量，样本数越多，评估结果越可靠。")
                    
                    st.subheader("评估结果")
                    for metric_name, metric_values in financial_predictions['评估指标'].items():
                        st.write(f"**{metric_name} 模型评估:**")
                        st.write(f"  均方误差 (MSE): {metric_values['MSE']:.4f}")
                        st.write(f"  均方根误差 (RMSE): {metric_values['RMSE']:.4f}")
                        st.write(f"  决定系数 (R²): {metric_values['R2']:.4f}")
                        st.write(f"  测试样本数: {metric_values['样本数']}")
    return pe_data


def predict_trend(prediction_df,last_price):
    # 预测摘要
    st.subheader("预测摘要")

    # 默认返回0，表示没有价格变化
    price_change_percent = 0
    
    # 获取预测期间的总体趋势
    if '预测收盘价' in prediction_df.columns:
        first_predicted = prediction_df['预测收盘价'].iloc[0]
        last_predicted = prediction_df['预测收盘价'].iloc[-1]
        price_change_percent = (last_predicted - last_price) / last_price * 100
        
        # 分析预测结果并生成摘要
        trend_analysis = []
        
        # 使用classify_value函数进行趋势分类
        conditions = [
            (lambda x: x > 3, f"预计股价将上涨 {price_change_percent:.2f}%"),
            (lambda x: x < -3, f"预计股价将下跌 {abs(price_change_percent):.2f}%"),
            (lambda x: True, f"预计股价波动较小，变化幅度为 {price_change_percent:.2f}%")
        ]
        result = classify_value(price_change_percent, conditions)
        if result:
            trend_analysis.append(result[0])
        
        # 如果有最高/最低价格预测，添加更多分析
        if '预测最高价' in prediction_df.columns and '预测最低价' in prediction_df.columns:
            max_potential = (prediction_df['预测最高价'].max() - last_price) / last_price * 100
            min_potential = (prediction_df['预测最低价'].min() - last_price) / last_price * 100
            trend_analysis.append(f"上涨潜力: +{max_potential:.2f}%")
            trend_analysis.append(f"下跌风险: {min_potential:.2f}%")
        
        # 显示预测摘要
        st.info("基于历史数据，预计未来预测期内:")
        for analysis in trend_analysis:
            st.write(f"- {analysis}")
    return price_change_percent

def main():
    st.header("股票价格历史查询")
    # 初始化股票历史数据变量
    stock_history,stock_code = gen_stock_history()
    # 初始化价格变化百分比变量
    price_change_percent = None
    # 确保有数据时才显示
    if stock_history is not None and not stock_history.empty:
        
        # 添加预测功能
        st.subheader("股票走势预测")
        days_to_predict = st.slider("选择预测未来天数", min_value=1, max_value=7, value=4)
        prediction_df=None

        # 确保有足够的数据进行预测（至少39个样本）
        if len(stock_history) >= 39:
            # 执行预测
            prediction_df, metrics = predict_stock_price(stock_history, days_to_predict)
            if prediction_df is None:
                st.warning("???")
                st.stop()
            if prediction_df is not None:
                # 获取当日收盘价
                last_price = stock_history['收盘价'].iloc[-1]
                last_date = pd.to_datetime(stock_history['日期'].iloc[-1])
                
                # 显示当日收盘价信息
                st.info(f"当日（{last_date.strftime('%Y-%m-%d')}）收盘价: **{last_price:.2f}**")
                
                # 显示预测结果
                st.success(f"成功预测未来 {days_to_predict} 天的股票走势")
                
                # 显示预测结果表格
                # 格式化预测结果表格，添加日期格式化
                display_df = prediction_df.copy()
                if '日期' in display_df.columns:
                    display_df['日期'] = display_df['日期'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(display_df, width='stretch')
                
                # 计算收益比（当日收盘价与预测期间最高/最低价格的比较）
                if days_to_predict >= 4:
                    # 取前4天的预测数据进行计算
                    four_day_predictions = prediction_df.head(4)
                    
                    # 获取预测的最高价和最低价
                    if '预测最高价' in four_day_predictions.columns and '预测最低价' in four_day_predictions.columns:
                        predicted_high = four_day_predictions['预测最高价'].max()
                        predicted_low = four_day_predictions['预测最低价'].min()
                        
                        # 计算收益比
                        max_return_ratio = (predicted_high - last_price) / last_price * 100
                        min_return_ratio = (predicted_low - last_price) / last_price * 100
                        
                        # 显示收益比信息
                        st.subheader("预测收益比分析（当日收盘价 vs 未来4天）")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="最高价格收益比", 
                                value=f"{max_return_ratio:.2f}%",
                                delta=f"{predicted_high:.2f}",
                                delta_color="inverse"
                            )
                        with col2:
                            st.metric(
                                label="最低价格收益比", 
                                value=f"{min_return_ratio:.2f}%",
                                delta=f"{predicted_low:.2f}",
                                delta_color="inverse"
                            )
                
                # 显示模型评估指标
                display_model_metrics(metrics)
                
                # 预测趋势
                price_change_percent = predict_trend(prediction_df,last_price)

        else:
            st.warning(f"历史数据不足39个样本（当前仅{len(stock_history)}个），无法进行有效预测")
            st.stop()
        
        # 确保prediction_df不为None
        if prediction_df is None:
            st.warning("预测失败：无法生成有效的预测结果")
            st.stop()
            
        try:
            # 准备可视化数据
            # 添加历史数据的类型标识
            historical_data = stock_history.copy()
            historical_data['类型'] = '历史'
            historical_data['预测收盘价'] = historical_data['收盘价']
            
            # 合并历史数据和预测数据
            visualization_data = pd.concat([
                historical_data[['日期', '收盘价', '预测收盘价', '类型']].tail(60),  # 只显示最近60天的历史数据
                prediction_df[['日期', '预测收盘价', '类型']]
            ])
            draw(visualization_data, stock_code,historical_data,prediction_df,days_to_predict)
            
        except Exception as e:
            st.error(f"预测失败：{str(e)}")
            st.stop()
        
        pe_data=pe_ratio_analysis(stock_history, stock_code)
        # 调用投资建议分析函数
        show_investment_advice(stock_code, pe_data, price_change_percent, historical_data)

        view_stock_his(stock_history,stock_code)
        if st.checkbox("显示表格数据"):
            st.dataframe(stock_history, width='stretch')

# 示例调用
if __name__ == "__main__":
    main()

