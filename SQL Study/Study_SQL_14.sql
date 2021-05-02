show databases

use world

select *
from city

select *
from city
where district = '' or district like '_'

select countrycode, count(countrycode), sum(population), avg(population)
from city
group by countrycode