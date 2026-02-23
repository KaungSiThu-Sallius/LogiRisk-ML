from pydantic import BaseModel, Field

class ShipmentInput(BaseModel):
    type: str
    days_for_shipment_scheduled: int
    category_id: int
    customer_segment: str
    department_id: int
    market: str
    order_item_quantity: int = Field(..., gt=0, description="Quantity must be at least 1")
    product_price: float = Field(..., gt=0, description="Price must be positive")
    shipping_mode: str
    order_city: str
    order_date: str

    class Config:
        populate_by_name = True