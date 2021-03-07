show databases

select *
from countrylanguage
where percentage >= 
(
select percentage
from countrylanguage
where countrycode = 
(
select distinct(countrycode)
from city
where countrycode = 'kor' and population > 1000000
) and isofficial = 'T'
)
order by percentage desc



