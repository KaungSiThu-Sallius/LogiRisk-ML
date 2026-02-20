import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
import joblib

def preprocess(dataset):
    root_path = os.getcwd()
    raw_dataset = 'data/raw/'+dataset
    df = pd.read_csv(os.path.join(root_path, raw_dataset), encoding='latin1', header=0)

    rm_cols = [ "Category Name", "Customer Id", "Customer City", "Customer Email", "Customer Fname", "Customer Lname", "Customer Password", "Customer Street", "Customer State",
            "Department Name", "Order Id", "Order Customer Id", "Order Item Cardprod Id", "Order Item Discount Rate", "Order Item Id", "Product Card Id",
            "Product Description", "Product Image", "Product Name", "Order Zipcode", "Days for shipping (real)", 'Delivery Status', "Order Status"]
    
    df_filtered = df.drop(rm_cols, axis=1)

    df_filtered['order date (DateOrders)'] = pd.to_datetime(
        df_filtered['order date (DateOrders)'], 
        format="%m/%d/%Y %H:%M"
    )
    df_filtered['shipping date (DateOrders)'] = pd.to_datetime(
        df_filtered['shipping date (DateOrders)'],
        format="%m/%d/%Y %H:%M"
    )

    rename_cols = {
    "Type": "type",
    "Days for shipment (scheduled)": "days_for_shipment_scheduled",
    "Benefit per order": "benefit_per_order",
    "Sales per customer": "sales_per_customer",
    "Late_delivery_risk": "late_delivery_risk",
    "Category Id": "category_id",
    "Customer Country": "customer_country",
    "Customer Segment": "customer_segment",
    "Customer Zipcode": "customer_zipcode",
    "Department Id": "department_id",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Market": "market",
    "Order City": "order_city",
    "Order Country": "order_country",
    "order date (DateOrders)": "order_date",
    "Order Item Discount": "order_item_discount",
    "Order Item Product Price": "order_item_product_price",
    "Order Item Profit Ratio": "order_item_profit_ratio",
    "Order Item Quantity": "order_item_quantity",
    "Sales": "sales",
    "Order Item Total": "order_item_total",
    "Order Profit Per Order": "order_profit_per_order",
    "Order Region": "order_region",
    "Order State": "order_state",
    "Product Category Id": "product_category_id",
    "Product Price": "product_price",
    "Product Status": "product_status",
    "shipping date (DateOrders)": "shipping_date",
    "Shipping Mode": "shipping_mode"
    }
    df_filtered.rename(columns=rename_cols, inplace=True)

    df_filtered['order_month'] = df_filtered['order_date'].dt.month
    df_filtered['shipping_month'] = df_filtered['shipping_date'].dt.month

    df_filtered['order_day'] = df_filtered['order_date'].dt.day_name()
    df_filtered['shipping_day'] = df_filtered['shipping_date'].dt.day_name()

    df_filtered = df_filtered.drop(['order_date', 'shipping_date'], axis=1)
    df_filtered = df_filtered.dropna()

    df_filtered.to_csv(os.path.join(root_path, 'data/processed/clean_dataset.csv'), index=False)

    print("Dataset is processed and save as CSV.")
    return df_filtered


def train_test_split_encode(df):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ordinal_encoder = OrdinalEncoder(categories=[days, days])
    df[['order_day', 'shipping_day']] = ordinal_encoder.fit_transform(df[['order_day', 'shipping_day']])

    root_path = os.getcwd()
    X = df.drop('late_delivery_risk', axis=1)
    y = df['late_delivery_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

    categorical_columns = X_train.select_dtypes(include=['object', 'string', 'category']).columns.to_list()

    target_encoder = TargetEncoder(cols=categorical_columns, smoothing=10.0)

    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)

    joblib.dump(target_encoder, os.path.join(root_path, 'models/target_encoder_v1.joblib'))
    joblib.dump(ordinal_encoder, os.path.join(root_path, 'models/ordinal_encoder_v1.joblib'))

    print("Dataframe is splited into train and test sets.")
    return (X_train_encoded, X_test_encoded, y_train, y_test)

# df = preprocess('DataCoSupplyChainDataset.csv')
# train_test_split_encode(df)