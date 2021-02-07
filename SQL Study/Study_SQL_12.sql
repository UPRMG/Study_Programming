show databases
use world



select countrycode, district, name, population
from city 
join 
(select max(population) as pol
from city
group by countrycode) as b
on city.population = b.pol



select countrycode, sum(population) as pol
from city
group by countrycode
order by pol desc
