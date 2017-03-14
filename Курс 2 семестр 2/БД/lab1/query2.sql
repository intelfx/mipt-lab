select
	distinct on(item.product_id) item.actual_price
from
	item
	join sales_order using(order_id)
	join customer using(customer_id)
where
	customer.name = 'WOMENS SPORTS'
order by
	item.product_id, sales_order.order_date asc;
