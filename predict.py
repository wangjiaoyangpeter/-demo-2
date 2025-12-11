from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from datetime import  timedelta


def generate_trading_dates(last_date, days_to_generate):
    """
    生成交易日历，排除周末，设置时间为15点
    
    :param last_date: 最后一个交易日期
    :param days_to_generate: 需要生成的交易天数
    :return: 交易日历列表
    """
    trading_dates = []
    current_date = last_date
    count = 0
    
    while count < days_to_generate:
        current_date += timedelta(days=1)
        # 排除周末（周六=5，周日=6）
        if current_date.weekday() < 5:
            # 设置时间为15:00
            trading_date = current_date.replace(hour=15, minute=0, second=0, microsecond=0)
            trading_dates.append(trading_date)
            count += 1
    
    return trading_dates

def predict_future_prices(model, scaler, last_features_scaled, days_to_predict):
    """
    预测未来价格
    :param model: 训练好的模型
    :param scaler: 用于特征缩放的MinMaxScaler
    :param last_features_scaled: 最新特征（已缩放）
    :param days_to_predict: 预测未来天数
    :return: 预测价格列表
    """
    predictions = []
    
    # 先获取原始特征（未缩放）
    last_features = scaler.inverse_transform(last_features_scaled)
    
    # 预测未来价格
    for _ in range(days_to_predict):
        # 预测下一天价格
        next_pred = model.predict(last_features_scaled)[0]
        predictions.append(next_pred)
        
        # 更新特征（将最新预测值加入特征的最前面，其他特征依次后移）
        # 确保new_features是一维数组
        new_features = np.roll(last_features[0], shift=1)  # 使用last_features[0]获取一维数组
        new_features[0] = next_pred  # 更新最前面的特征为最新预测
        
        # 更新移动平均（简化处理）- 确保计算正确且结果为标量
        recent_predictions = predictions[-5:] if len(predictions)>=5 else predictions + [last_features[0, 0]]*(5-len(predictions))
        new_features[-2] = np.mean(recent_predictions)
        
        recent_predictions_20 = predictions[-min(len(predictions), 20):]
        if len(recent_predictions_20) < 20:
            recent_predictions_20 = recent_predictions_20 + [last_features[0, 0]]*(20-len(recent_predictions_20))
        new_features[-1] = np.mean(recent_predictions_20)
        
        last_features = new_features.reshape(1, -1)  # 重塑为二维数组
        last_features_scaled = scaler.transform(last_features)
    return predictions
            

def predict_stock_price(stock_history, days_to_predict=5, feature_days=10):  # 减少feature_days以避免数据稀疏
    """
    预测股票未来价格（包括收盘价、最高价、最低价）
    :param stock_history: 历史股票数据
    :param days_to_predict: 预测未来天数
    :param feature_days: 用于预测的历史天数
    :return: 预测结果DataFrame和模型评估指标
    """

    try:
        # 创建特征和标签
        df = stock_history.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 创建用于预测的所有目标变量
        targets = ['收盘价', '最高价', '最低价']
        all_predictions = {}
        all_metrics = {}
        
        # 为每个目标变量分别训练模型并进行预测
        for target in targets:
            if target not in df.columns:
                st.warning(f"缺少{target}数据，跳过该指标预测")
                continue
                
            # 使用当前目标变量作为预测目标
            data = df[['日期', target]].copy()
            
            # 确保目标变量是数值类型
            data[target] = pd.to_numeric(data[target], errors='coerce')
            
            # 创建时间序列特征 - 使用较少的天数以避免数据稀疏
            for i in range(1, min(feature_days, 10) + 1):  # 最多使用10个lag特征
                data[f'{target}_lag_{i}'] = data[target].shift(i)
            
            # 创建移动平均特征
            data['ma5'] = data[target].rolling(window=5).mean()
            data['ma20'] = data[target].rolling(window=20).mean()
            
            # 删除包含NaN的行
            data = data.dropna()
            
            if len(data) < 20:  # 数据不足时返回
                return None, {"error": f"{target}历史数据不足，无法进行预测"}
            
            # 准备特征和目标变量
            X = data.drop(['日期', target], axis=1)
            y = data[target]
            
            # 确保特征数据是数值类型且形状一致
            X = X.astype(float)
            
            # 数据归一化
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X.values)  # 使用.values确保是numpy数组
            
            # 划分训练集和测试集
            train_size = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # 训练线性回归模型
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 预测测试集
            y_pred = model.predict(X_test)
            
            # 计算模型评估指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            all_metrics[target] = {
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            # 准备预测未来数据
            last_features = X.iloc[-1].values.reshape(1, -1)
            last_features_scaled = scaler.transform(last_features)
            
            predictions = []
            
            predictions = predict_future_prices(model, scaler, last_features_scaled, days_to_predict)
            
            all_predictions[target] = predictions
        
        # 生成预测日期（交易日）
        last_date = df['日期'].iloc[-1]
        forecast_dates = generate_trading_dates(last_date, days_to_predict)
        
        # 创建预测结果DataFrame
        prediction_df = pd.DataFrame({
            '日期': forecast_dates
        })
        
        # 添加预测数据
        for target, preds in all_predictions.items():
            prediction_df[f'预测{target}'] = preds
        
        prediction_df['类型'] = '预测'
        
        # 合并所有指标的评估结果
        overall_metrics = {
            'mse': {target: metrics['mse'] for target, metrics in all_metrics.items()},
            'rmse': {target: metrics['rmse'] for target, metrics in all_metrics.items()},
            'r2': {target: metrics['r2'] for target, metrics in all_metrics.items()},
            'train_samples': next(iter(all_metrics.values()))['train_samples'] if all_metrics else 0,
            'test_samples': next(iter(all_metrics.values()))['test_samples'] if all_metrics else 0
        }
        
        return prediction_df, overall_metrics
    
    except Exception as e:
        return None, {"error": f"预测失败: {str(e)}"}

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
        last_date = pd.to_datetime(pe_data['日期'].iloc[-1])
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