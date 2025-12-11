import altair as alt
import streamlit as st
import pandas as pd

def view_stock_his(stock_history,stock_code):
    # 原有的历史数据可视化保持不变
    st.subheader("历史数据可视化")
    
    # 设置Altair图表的默认设置，确保显示所有数据点
    alt.data_transformers.disable_max_rows()
    
    # 创建基础图表
    base = alt.Chart(stock_history, width=800, height=600)
    
    # 创建选择器用于交互
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['日期'], empty='none')
    
    choices=st.multiselect("请选择要显示的指标", ['开盘价', '收盘价', '最高价', '最低价'], 
        default=['开盘价', '收盘价', '最高价', '最低价'])
    
    # 绘制多条线
    lines = base.mark_line().encode(
        x='日期:T',
        y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
        color='variable:N',
        strokeWidth=alt.value(2)
    ).transform_fold(
        choices,
        as_=['variable', 'value']
    )
    
    # 添加点和工具提示
    points = lines.mark_point().encode(
        opacity=alt.value(0)
    ).add_params(
        nearest
    )
    
    # 添加垂直线和交互式工具提示
    rule = base.mark_rule(color='gray').encode(
        x='日期:T',
    ).transform_filter(
        nearest
    )
    
    # 添加详细的工具提示
    tooltip = alt.Chart(stock_history).mark_text(align='left', dx=5, dy=-5).encode(
        x='日期:T',
        y='value:Q',
        text=alt.Text('value:Q', format='.2f'),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        color='variable:N'
    ).transform_fold(
        ['开盘价', '收盘价', '最高价', '最低价'],
        as_=['variable', 'value']
    ).add_params(
        nearest
    )
    
    # 组合所有图层
    chart = alt.layer(
        lines,
        points,
        rule,
        tooltip
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        gridOpacity=0.3
    ).properties(
        title=f'{stock_code} 股票价格历史走势图'
    ).interactive()
    
    st.altair_chart(chart, width='stretch')
        
        

def view_pe(pe_data,stock_code):
    st.write("### 市盈率(PE)分析")    
    # 显示PE数据摘要
    latest_pe = pe_data['市盈率(PE)'].iloc[-1]
    avg_pe = pe_data['市盈率(PE)'].mean()
    st.metric("最新市盈率", f"{latest_pe:.2f}", f"{latest_pe - avg_pe:.2f}")
    
    # PE走势图
    pe_chart = alt.Chart(pe_data).mark_line(color='#2ca02c', strokeWidth=2).encode(
        x='日期:T',
        y=alt.Y('市盈率(PE):Q', scale=alt.Scale(zero=False), title='市盈率'),
        tooltip=[alt.Tooltip('日期:T', format='%Y-%m-%d'),
                    alt.Tooltip('市盈率(PE):Q', format='.2f')]
    ).properties(
        title=f'{stock_code} 市盈率走势图',
        width='container',
        height=400
    ).interactive()
    st.altair_chart(pe_chart, use_container_width=True)

def draw(visualization_data, stock_code,historical_data,prediction_df,days_to_predict):
    # 创建预测图表
    st.subheader("历史价格与预测走势")
    
    # 基础图表设置
    prediction_base = alt.Chart(visualization_data, width=800, height=600)
    
    # 绘制历史价格线
    historical_line = prediction_base.mark_line(color='#1f77b4', strokeWidth=2).encode(
        x='日期:T',
        y=alt.Y('预测收盘价:Q', scale=alt.Scale(zero=False), title='价格'),
        tooltip=[alt.Tooltip('日期:T', format='%Y-%m-%d'), 
                    alt.Tooltip('预测收盘价:Q', format='.2f', title='价格'),
                    alt.Tooltip('类型:N')]
    ).transform_filter(
        alt.datum.类型 == '历史'
    )
    
    # 绘制预测价格线（虚线）
    prediction_line = prediction_base.mark_line(color='#ff7f0e', strokeDash=[5, 5], strokeWidth=2).encode(
        x='日期:T',
        y='预测收盘价:Q',
        tooltip=[alt.Tooltip('日期:T', format='%Y-%m-%d'), 
                    alt.Tooltip('预测收盘价:Q', format='.2f', title='预测价格'),
                    alt.Tooltip('类型:N')]
    ).transform_filter(
        alt.datum.类型 == '预测'
    )
    
    # 添加预测点
    prediction_points = prediction_base.mark_point(color='#ff7f0e', size=60).encode(
        x='日期:T',
        y='预测收盘价:Q',
        tooltip=[alt.Tooltip('日期:T', format='%Y-%m-%d'), 
                    alt.Tooltip('预测收盘价:Q', format='.2f', title='预测价格'),
                    alt.Tooltip('类型:N')]
    ).transform_filter(
        alt.datum.类型 == '预测'
    )
    
    # 添加分割线标识预测开始位置
    last_historical_date = historical_data['日期'].max()
    vertical_line = alt.Chart(pd.DataFrame({'x': [last_historical_date]})).mark_rule(
        color='gray',
        strokeDash=[5, 5],
        strokeWidth=1
    ).encode(x='x:T')
    
    # 添加图例说明
    legend_data = pd.DataFrame([
        {'category': '历史价格', 'color': '#1f77b4'},
        {'category': '预测价格', 'color': '#ff7f0e'}
    ])
    
    legend = alt.Chart(legend_data).mark_rect().encode(
        y=alt.Y('category:N', axis=alt.Axis(orient='right')),
        color=alt.Color('color:N', scale=None)
    )
    
    # 组合所有图层
    prediction_chart = alt.layer(
        historical_line,
        prediction_line,
        prediction_points,
        vertical_line
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        gridOpacity=0.3
    ).properties(
        title=f'{stock_code} 股票价格历史与未来预测走势',
        width='container',
        height=500
    ).interactive()
    
    st.altair_chart(prediction_chart, use_container_width=True)
    
    # 计算预测趋势
    if '收盘价' in historical_data.columns and '预测收盘价' in prediction_df.columns:
        last_historical_price = historical_data['收盘价'].iloc[-1]
        last_prediction_price = prediction_df['预测收盘价'].iloc[-1]
        price_change_percent = ((last_prediction_price - last_historical_price) / last_historical_price) * 100
    else:
        st.warning("缺少必要的价格数据，无法计算预测趋势")
        return
    
    # 显示预测摘要
    st.info(f"预测摘要: 基于历史数据，预计未来 {days_to_predict} 天内，股价可能{'' if price_change_percent >= 0 else '下跌'} {abs(price_change_percent):.2f}%")

def draw_pe(pred_df, pred_results, pred_dates, stock_code):
    # 添加参数检查
    if pred_results is None:
        st.warning("没有可用的市盈率预测数据")
        return
    
    if '市盈率(PE)' not in pred_results:
        st.warning("预测结果中缺少市盈率数据")
        return
    
    pred_df['预测PE'] = pred_results['市盈率(PE)']
    st.write("##### 市盈率预测")
    
    # PE预测图表
    pe_pred_data = pd.DataFrame({
        '日期': pred_dates,
        '预测PE': pred_results['市盈率(PE)']
    })
    
    pe_pred_chart = alt.Chart(pe_pred_data).mark_line(color='#2ca02c', strokeDash=[5, 5], strokeWidth=2).encode(
        x='日期:T',
        y=alt.Y('预测PE:Q', scale=alt.Scale(zero=False), title='预测市盈率'),
        tooltip=[alt.Tooltip('日期:T', format='%Y-%m-%d'),
                    alt.Tooltip('预测PE:Q', format='.2f')]
    ).properties(
        title=f'{stock_code} 市盈率预测走势',
        width='container',
        height=300
    ).interactive()
    st.altair_chart(pe_pred_chart, use_container_width=True)

def display_model_metrics(metrics):
    with st.expander("查看模型评估指标"):
        # 模型评估指标说明
        st.write("### 指标说明\n- **均方误差 (MSE)**: 衡量预测值与实际值之间的平均平方误差，值越小表示模型拟合越好")
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
