select * 
from dept_master


select *
from emp_master


select *
from address_master


select gt.address_id, concat(gt.city, ' ',gt.gu, ' ', gt.address_name) as concat,
bt.emp_name, bt.age, bt.dept_id
from address_master as gt, emp_master as bt
where gt.address_id = bt.address_id


select c.address_id, c.emp_name, a.dept_name, c.concat
from dept_master as a, 
(
select gt.address_id, concat(gt.city, ' ',gt.gu, ' ', gt.address_name) as concat,
bt.emp_name, bt.age, bt.dept_id
from address_master as gt, emp_master as bt
where gt.address_id = bt.address_id
) as c
where c.dept_id = a.dept_id






