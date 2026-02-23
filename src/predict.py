import joblib
from datetime import datetime
import os
import pandas as pd
from xgboost import XGBClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogiRisk_Predictor:
    def __init__(self, model_dir = 'models'):
        try: 
            models_path = os.path.join(os.getcwd(), model_dir)
            self.model = XGBClassifier()
            self.model.load_model(os.path.join(models_path, 'best_xgb_model_logi_risk.json'))

            self.ordinal_encoder = joblib.load(os.path.join(models_path, 'ordinal_encoder_v1.joblib'))
            self.target_encoder = joblib.load(os.path.join(models_path, 'target_encoder_v1.joblib'))

            self.feature_columns = [
                'type','days_for_shipment_scheduled','category_id','customer_segment','department_id',
                'market','order_item_quantity','product_price','shipping_mode','order_address', 'urgency_score', 'order_month', 'order_day'
            ]

            logger.info("All models and encoders have loaded successfully")
        except Exception as e:
            logger.error("Failed to load the artifacts {e}")
            raise
        
    def _engineer_features(self, df):
        df['order_address'] = df['order_city'] + '_' + df['shipping_mode']
        df['urgency_score'] = df['days_for_shipment_scheduled']/df['order_item_quantity']

        df['order_date'] = pd.to_datetime(df['order_date'])
        df['order_month'] = df['order_date'].dt.month
        df['order_day'] = df['order_date'].dt.day_name()

        return df.drop(['order_city', 'order_date'], axis=1)

    def predict(self, raw_data: dict):
        df = pd.DataFrame(raw_data)
        df_filtered = self._engineer_features(df)

        df_filtered[['order_day']] = self.ordinal_encoder.transform(df_filtered[['order_day']])

        df_filtered = self.target_encoder.transform(df_filtered)

        df_filtered = df_filtered[self.feature_columns]

        risk_probability = self.model.predict_proba(df_filtered)[0][1]
        
        return float(risk_probability)
    
if __name__ == '__main__':
    predictor = LogiRisk_Predictor()

    sample_input = {
        "type": "TRANSFER",
        "days_for_shipment_scheduled": 4,
        "category_id": 73,
        "customer_segment": "Consumer",
        "department_id": 2,
        "market": "Pacific Asia",
        "order_item_quantity": 1,
        "product_price": 327.75,
        "shipping_mode": "Standard Class",
        "order_city": "Bekasi",
        "order_date": "2026-02-22"
    }


    sample_result = predictor.predict([sample_input])

    print(f"Risk Probability: {sample_result:.2%}")


