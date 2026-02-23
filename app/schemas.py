from pydantic import BaseModel, Field

class ShipmentInput(BaseModel):
    type: str
    days_for_shipment_scheduled: int
    category_id: int
    customer_segment: str
    department_id: int
    market: str
    order_item_quantity: int
    product_price: float
    shipping_mode: str
    order_city: str
    order_date: str