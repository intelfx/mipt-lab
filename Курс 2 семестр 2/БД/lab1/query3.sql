/* compute employees' income per month, taking into account their per-order commission rate */
with employee_income as (
	select
		employee.*,
		case
		when
			employee.commission > 0
			and count(sales_order.order_id) > 0
		then
			employee.salary + cast(employee.commission as double precision) * count(sales_order.order_id) / count(distinct to_char(sales_order.order_date, 'YYYY-MM'))
		else
			employee.salary
		end as income
	from
		employee
		left join customer on employee.employee_id = customer.salesperson_id
		left join sales_order using(customer_id)
	group by employee.employee_id
)

/* group by department, pick one with greatest average income */
select
	department.department_id,
	avg(employee_income.income) as avg_income
from
	employee_income
	join department using(department_id)
group by department.department_id
order by avg_income desc
limit 1;
