show databases

use world

select * 
from city

-- find ID pus% or seoul
select ID
from city
where name like 'pus%' or name like 'seoul'

-- 2 condition
select * 
from city
where population > 500000 and id between 5 and 100

-- After group by countrycode sum population
select countrycode, sum(population) as sum_po
from city
group by countrycode
order by sum_po desc
limit 15