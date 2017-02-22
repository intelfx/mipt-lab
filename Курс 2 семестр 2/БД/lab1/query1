select
	avg(sales_order.total) as total,
	to_char(sales_order.order_date, 'YYYY-MM') as month
from
	sales_order
	join customer using(customer_id)
where
	customer.state = 'CA'
group by
	month;
