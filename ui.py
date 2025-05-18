import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide", page_title="سیستم تشخیص تقلب مالی")
pd.set_option("styler.render.max_elements", 500000)

st.title("سیستم تحلیل و تشخیص تقلب مالی")
st.markdown("""
این برنامه تمام مراحل تحلیل داده‌های تقلب مالی را از پیش‌پردازش تا مدل‌سازی پوشش می‌دهد.
""")

st.header("1. بارگذاری و پیش‌پردازش داده‌ها")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/fraud_dataset_mod.csv')
        
        df['Daily_Transaction_Count'] = df['Daily_Transaction_Count'].fillna(df['Daily_Transaction_Count'].mean())
        df['Transaction_Amount'] = df['Transaction_Amount'].fillna(df['Transaction_Amount'].median())
        
        categorical_cols = ['Is_Weekend', 'Account_Balance', 'Avg_Transaction_Amount_7d', 
                          'Failed_Transaction_Count_7d', 'Transaction_Distance', 'Risk_Score',
                          'Fraud_Label', 'Previous_Fraudulent_Activity', 'IP_Address_Flag',
                          'Card_Age']
        
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df
    except Exception as e:
        st.error(f"خطا در بارگذاری داده‌ها: {str(e)}")
        return None

df = load_data()
if df is None:
    st.stop()

if st.checkbox("نمایش داده‌های خام"):
    st.dataframe(df)

st.header("2. تحلیل اکتشافی داده‌ها")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

tab1, tab2, tab3 = st.tabs(["توزیع داده‌های عددی", "داده‌های پرت", "تحلیل دسته‌ای"])

with tab1:
    st.subheader("توزیع داده‌های عددی")
    selected_num_col = st.selectbox("ستون عددی را انتخاب کنید", numeric_cols, key="num_col")
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(df[selected_num_col], kde=True, ax=ax[0])
    ax[0].set_title(f'Histogram of {selected_num_col}')
    
    sns.boxplot(x=df[selected_num_col], ax=ax[1])
    ax[1].set_title(f'Boxplot of {selected_num_col}')
    st.pyplot(fig)

with tab2:
    st.subheader("شناسایی داده‌های پرت")
    method = st.radio("روش شناسایی", ["IQR", "Z-Score"], horizontal=True)
    outlier_col = st.selectbox("ستون مورد نظر", numeric_cols, key="outlier_col")
    
    if method == "IQR":
        Q1 = df[outlier_col].quantile(0.25)
        Q3 = df[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
        
        st.write(f"**نتایج روش IQR برای ستون {outlier_col}**")
        st.write(f"- Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        st.write(f"- حد پایین: {lower_bound:.2f}, حد بالا: {upper_bound:.2f}")
        st.write(f"- تعداد داده‌های پرت: {len(outliers)}")
        
    else:  # Z-Score
        mean_val = df[outlier_col].mean()
        std_val = df[outlier_col].std()
        df['Z_Score'] = np.abs((df[outlier_col] - mean_val) / std_val)
        outliers = df[df['Z_Score'] > 3]
        
        st.write(f"**نتایج روش Z-Score برای ستون {outlier_col}**")
        st.write(f"- میانگین: {mean_val:.2f}, انحراف معیار: {std_val:.2f}")
        st.write(f"- تعداد داده‌های پرت: {len(outliers)}")
        
    if st.checkbox("نمایش نمونه‌ای از داده‌های پرت"):
        st.dataframe(outliers.head())

with tab3:
    st.subheader("تحلیل داده‌های دسته‌ای")
    selected_cat_col = st.selectbox("ستون دسته‌ای را انتخاب کنید", categorical_cols, key="cat_col")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=selected_cat_col, hue="Fraud_Label", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.header("3. مهندسی ویژگی‌ها")

if st.checkbox("انجام مهندسی ویژگی‌ها"):
    df['Log_Transaction_Amount'] = np.log1p(df['Transaction_Amount'])
    df['Log_Account_Balance'] = np.log1p(df['Account_Balance'])
    
    cols_to_drop = st.multiselect(
        df.columns.tolist(),
        default=['Transaction_ID','User_ID','Transaction_Type','Timestamp',
               'Device_Type','Location','Merchant_Category','Card_Type',
               'Authentication_Method']
    )
    
    df = df.drop(cols_to_drop, axis=1)
    st.success("مهندسی ویژگی‌ها با موفقیت انجام شد!")
    st.dataframe(df.head())

st.header("4. مدل‌سازی و ارزیابی")

X = df.drop(columns=['Fraud_Label', 'Normalized_Amount_Std', 'Amount_Median_Diff'])
y = df['Fraud_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_type = st.selectbox("انتخاب مدل", ["Random Forest", "XGBoost"])

if st.button("آموزش و ارزیابی مدل"):
    with st.spinner('در حال آموزش مدل...'):
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = XGBClassifier(random_state=42)
            
        model.fit(X_train, y_train)
        
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        
        st.subheader("نتایج ارزیابی مدل")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**گزارش طبقه‌بندی:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.table(pd.DataFrame(report).transpose())
            
        with col2:
            st.write("**ماتریس درهم‌ریختگی:**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('پیش‌بینی شده')
            ax.set_ylabel('واقعی')
            st.pyplot(fig)
        
        
        st.write("**منحنی ROC:**")
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        
        st.write("**اهمیت ویژگی‌ها:**")
        
        if model_type == "Random Forest":
            importance = pd.Series(model.feature_importances_, index=X.columns)
        else:
            importance = pd.Series(model.feature_importances_, index=X.columns)
            
        importance = importance.sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        
        joblib.dump(model, 'fraud_model.joblib')
        st.success("مدل با موفقیت ذخیره شد!")


st.header("5. پیش‌بینی تراکنش جدید")

if st.checkbox("بارگذاری مدل برای پیش‌بینی جدید"):
    try:
        model = joblib.load('fraud_model.joblib')
        st.success("مدل با موفقیت بارگذاری شد!")
        
        
        with st.form("prediction_form"):
            st.write("مشخصات تراکنش جدید را وارد کنید:")
            
            input_data = {}
            for col in X.columns:
                if df[col].dtype in ['int64', 'float64']:
                    input_data[col] = st.number_input(col, value=df[col].median())
                else:
                    input_data[col] = st.selectbox(col, options=df[col].unique())
            
            submitted = st.form_submit_button("پیش‌بینی")
            if submitted:
                input_df = pd.DataFrame([input_data])
                
                
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)[0][1]
                
                if prediction[0] == 1:
                    st.error(f"⚠️ این تراکنش مشکوک به تقلب است (احتمال: {probability:.2%})")
                else:
                    st.success(f"✅ این تراکنش عادی است (احتمال: {1-probability:.2%})")
                    
    except:
        st.error("مدل یافت نشد! لطفاً ابتدا مدل را آموزش دهید.")