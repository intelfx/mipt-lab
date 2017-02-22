/* compute managers' salary per inferior */
with manager_stats as (
	select
		manager.*,
		manager.salary / count(inferior.employee_id) as salary_per_inferior
	from
		employee as manager
		join employee as inferior on manager.employee_id = inferior.manager_id
	group by manager.employee_id
)

/* group globally, compute average */
select
	avg(manager_stats.salary_per_inferior) as avg_salary_per_inferior
from
	manager_stats;
