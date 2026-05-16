import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option('display.max_columns', None)


def clean_bedrooms(df):
    df_clean = df.copy()
    # 转为字符串并小写
    df_clean['Bedrooms_str'] = df_clean['Bedrooms'].astype(str).str.lower()

    # 1. 提取数字
    df_clean['Bedrooms_Count'] = df_clean['Bedrooms_str'].str.extract(r'(\d+)').astype(float)

    # 2. 提取高级卖点 (创建特征列)
    df_clean['Has_Master_Suite'] = df_clean['Bedrooms_str'].str.contains('master suite|master retreat',na=False).astype(int)
    df_clean['Has_Ground_Bed'] = df_clean['Bedrooms_str'].str.contains('ground', na=False).astype(int)
    df_clean['Has_Walkin_Closet'] = df_clean['Bedrooms_str'].str.contains('walk-in|walk in', na=False).astype(int)

    text_implies_bed = df_clean['Bedrooms_str'].str.contains('bed|suite|retreat', na=False)
    is_count_nan = df_clean['Bedrooms_Count'].isna()

    implied_count = df_clean[['Has_Master_Suite', 'Has_Ground_Bed']].sum(axis=1)
    # 将加起来的值（最小为1）赋给那些有描述但没数字的行
    df_clean.loc[is_count_nan & text_implies_bed, 'Bedrooms_Count'] = implied_count.replace(0, 1)
    # ====================================================

    # 删掉原始脏乱的列和中转列
    df_clean = df_clean.drop(columns=['Bedrooms_str', 'Bedrooms'])

    return df_clean

def clean_time_features(df):
    df_clean = df.copy()

    # 1. 转换为 datetime 类型
    # errors='coerce' 意思是如果遇到无法解析的乱码时间，自动变成 NaT (缺失值)，防止代码报错崩溃
    df_clean['Listed On'] = pd.to_datetime(df_clean['Listed On'], errors='coerce')
    df_clean['Last Sold On'] = pd.to_datetime(df_clean['Last Sold On'], errors='coerce')

    # 2. 提取绝对时间特征
    df_clean['Listed_Year'] = df_clean['Listed On'].dt.year
    df_clean['Listed_Month'] = df_clean['Listed On'].dt.month

    # 3. 提取相对时间差
    # 挂牌距上次售出的天数
    df_clean['Days_Between_Sales'] = (df_clean['Listed On'] - df_clean['Last Sold On']).dt.days

    # 房龄：挂牌年份 - 建成年份
    if 'Year built' in df_clean.columns:
        df_clean['House_Age'] = df_clean['Listed_Year'] - df_clean['Year built']

    # 4. 删掉原始的 datetime 列 (因为模型无法直接吃 datetime 格式的数据)
    df_clean = df_clean.drop(columns=['Listed On', 'Last Sold On'])

    return df_clean

train_data=pd.read_csv('./test/train.csv')
test_data=pd.read_csv('./test/test.csv')

# 'Address'
location_cols = [
    'Region', 'City', 'Zip'
]

price_cols = [
    'Sold Price', 'Listed On', 'Listed Price', 'Last Sold On', 'Last Sold Price'
]

property_cols = [
    'Type', 'Year built', 'Lot', 'Total interior livable area',
    'Bedrooms', 'Bathrooms', 'Full bathrooms'
]

amenities_cols = [
    'Heating', 'Heating features', 'Cooling', 'Cooling features',
    'Parking', 'Total spaces', 'Garage spaces', 'Parking features',
    'Flooring', 'Appliances included', 'Laundry features'
]

school_cols = [
    'Elementary School', 'Elementary School Score', 'Elementary School Distance',
    'Middle School', 'Middle School Score', 'Middle School Distance',
    'High School', 'High School Score', 'High School Distance'
]

tax_cols = [
    'Tax assessed value', 'Annual tax amount'
]

all_cols_dict = {
    'location': location_cols,
    'price': price_cols,
    'property': property_cols,
    'amenities': amenities_cols,
    'school': school_cols,
    'tax': tax_cols
}

new_order = []

for cols in all_cols_dict.values():
    new_order.extend(cols)

train_data = train_data[new_order]

train_data=clean_bedrooms(train_data)
train_data=clean_time_features(train_data)

test_data = clean_bedrooms(test_data)
test_data = clean_time_features(test_data)

train_data.info()
print(train_data.iloc[1,:])

X=train_data.drop(columns=['Sold Price'])
y=train_data['Sold Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

# 1. 目标编码 (Target Encoding) - 适合类别种类极多的特征
target_encode_cols = [
    'Zip', 'City', 'Region',
    'Elementary School', 'Middle School', 'High School'
]

one_hot_cols = [
    'Type'
]

fill_none_cols = [
    'Cooling', 'Heating',
    'Heating features', 'Cooling features',
    'Parking', 'Parking features',
    'Flooring', 'Appliances included', 'Laundry features'
]

numeric_cols = [
    'Lot', 'Year built', 'Bathrooms', 'Bedrooms_Count',
    'Total interior livable area', 'Full bathrooms',
    'Tax assessed value', 'Annual tax amount',
    'Elementary School Score', 'Middle School Score', 'High School Score',
    'Listed Price', 'Last Sold Price',
    'Listed_Year', 'Listed_Month', 'Days_Between_Sales', 'House_Age',
    'Total spaces', 'Garage spaces',
    'Elementary School Distance', 'Middle School Distance', 'High School Distance',
    'Has_Master_Suite', 'Has_Ground_Bed', 'Has_Walkin_Closet'
]

preprocessor = ColumnTransformer(
    transformers=[
        # 1. 对高基数地区/学校，以及复杂的设施文本，统统使用 目标编码！
        # 这里用了一个内部 Pipeline：先填补缺失值，再做目标编码
        ('target_and_text', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
            ('te', TargetEncoder(smoothing=10))
        ]), target_encode_cols + fill_none_cols), # <--- 注意这里把两个列表加起来了

        # 2. 独热编码：只保留给真正简单的特征 (如 Type)
        ('onehot', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), one_hot_cols),

        # 3. 数值类：缺失填补为中位数
        ('numeric', SimpleImputer(strategy='median'), numeric_cols)
    ],
    remainder='drop'
)

X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)

# 1. 初始化模型
# n_estimators=100: 种植100棵决策树
# n_jobs=-1: 调用你电脑所有的 CPU 核心全速运算
# random_state=666: 保证每次运行的结果一致
print("正在初始化随机森林模型...")
model = RandomForestRegressor(n_estimators=100, random_state=666, n_jobs=-1)

# 2. 训练模型
print("模型开始训练，请稍候 (可能需要几十秒到一两分钟)...")
model.fit(X_train_processed, y_train)
print("模型训练完成！")

# 3. 让模型在它从未见过的测试集上做预测
print("正在进行价格预测...")
y_pred = model.predict(X_test_processed)

# 4. 评估模型表现 (看看它猜得有多准)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*20 + " 模型基准测试结果 " + "="*20)
print(f"平均绝对误差 (MAE): ${mae:,.2f}")
print(f"均方根误差 (RMSE): ${rmse:,.2f}")
print(f"决定系数 (R2 Score): {r2:.4f}")
print("="*58)

print("准备预测测试集...")

# 1. 获取测试集的 ID
raw_test_data = pd.read_csv('./test/test.csv')
test_ids = raw_test_data['Id']

# 确保你的 test_data 已经跑过了之前的清洗函数：
# test_data = clean_bedrooms(test_data)
# test_data = clean_time_features(test_data)
# test_data = test_data.drop(columns=['Address', 'Summary', 'Id'], errors='ignore')
# (注意：如果 test_data 里有之前没清干净的列，确保在这里处理掉，让它的列和训练时的 X 完全一致)

# 2. 将测试集送入流水线进行预处理 (严格使用 transform)
print("正在通过 Pipeline 预处理测试集特征...")
X_test_final = preprocessor.transform(test_data)

# 3. 使用训练好的随机森林模型进行预测
print("正在计算最终的预测价格...")
final_predictions = model.predict(X_test_final)

# 4. 组装输出结果
print("正在生成提交文件...")
submission = pd.DataFrame({
    'Id': test_ids,
    'Sold Price': final_predictions
})

# 5. 保存为 CSV 文件
# index=False 的意思是不要把 DataFrame 自带的行号(0, 1, 2...)存进去
submission.to_csv('my_baseline_submission.csv', index=False)

