import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
import joblib
import json


def preprocess(dataset):
    root_path = os.getcwd()
    raw_dataset = 'data/raw/'+dataset
    df = pd.read_csv(os.path.join(root_path, raw_dataset), encoding='latin1', header=0)

    # df['Customer Zipcode'] = df['Customer Zipcode'].astype('str')
    df['order_address'] = df['Order City']  + "_" + df['Shipping Mode']
    # df['customer_address'] = df['Customer Zipcode'] + "_" + df['Customer Country']

    rm_cols = [ "Category Name", "Customer Id", "Customer City", "Customer Email", "Customer Fname", "Customer Lname", "Customer Password", "Customer Street", "Customer State",
            "Department Name", "Order Id", "Order Customer Id", "Order Item Cardprod Id", "Order Item Discount Rate", "Order Item Id", "Product Card Id",
            "Product Description", "Product Image", "Product Name", "Order Zipcode", "Days for shipping (real)", 'Delivery Status', "Order Status", "shipping date (DateOrders)",
            "Order Item Profit Ratio", "Order Profit Per Order", "Sales per customer", "Order Item Total", "Benefit per order", "Sales", "Latitude", "Longitude",
            'Customer Zipcode', 'Customer Country', 'Order City', 'Order Country', 'Order State', 'Order Item Discount', 'Order Region']
    
    df_filtered = df.drop(rm_cols, axis=1)

    df_filtered['order date (DateOrders)'] = pd.to_datetime(
        df_filtered['order date (DateOrders)'], 
        format="%m/%d/%Y %H:%M"
    )


    rename_cols = {
    "Type": "type",
    "Days for shipment (scheduled)": "days_for_shipment_scheduled",
    "Late_delivery_risk": "late_delivery_risk",
    "Category Id": "category_id",
    "Customer Segment": "customer_segment",
    "Department Id": "department_id",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Market": "market",
    "order date (DateOrders)": "order_date",
    "Order Item Product Price": "order_item_product_price",
    "Order Item Quantity": "order_item_quantity",
    "Product Category Id": "product_category_id",
    "Product Price": "product_price",
    "Product Status": "product_status",
    "Shipping Mode": "shipping_mode"
    }
    df_filtered.rename(columns=rename_cols, inplace=True)

    df_filtered['urgency_score'] = df_filtered['days_for_shipment_scheduled'] / df_filtered['order_item_quantity']

    df_filtered['order_month'] = df_filtered['order_date'].dt.month

    df_filtered['order_day'] = df_filtered['order_date'].dt.day_name()

    df_filtered = df_filtered.drop(['order_date'], axis=1)
    df_filtered = df_filtered.dropna()

    df_filtered.to_csv(os.path.join(root_path, 'data/processed/clean_dataset.csv'), index=False)

    print("Dataset is processed and save as CSV.")
    return df_filtered


def train_test_split_encode(df):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ordinal_encoder = OrdinalEncoder(categories=[days])
    df[['order_day']] = ordinal_encoder.fit_transform(df[['order_day']])

    root_path = os.getcwd()
    X = df.drop('late_delivery_risk', axis=1)
    y = df['late_delivery_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

    categorical_columns = X_train.select_dtypes(include=['object', 'string', 'category']).columns.to_list()

    target_encoder = TargetEncoder(cols=categorical_columns, smoothing=20.0)

    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)

    joblib.dump(target_encoder, os.path.join(root_path, 'models/target_encoder_v1.joblib'))
    joblib.dump(ordinal_encoder, os.path.join(root_path, 'models/ordinal_encoder_v1.joblib'))

    print("Dataframe is splited into train and test sets.")
    return (X_train_encoded, X_test_encoded, y_train, y_test)

# df = preprocess('DataCoSupplyChainDataset.csv')
# train_test_split_encode(df)