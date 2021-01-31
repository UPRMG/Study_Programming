show databases

use world

select * 
from country

-- 여러조건을 만족하는 코드 추출
select code
from country
where headofstate like 'A%'
and governmentform = 'Republic'
and surfacearea > 500000 and population > 5000000
and continent = 'Asia'

-- 천만이상 국가의 gnp 비중 추출
select 
round((select sum(gnp)
from country
where population > 10000000) / sum(gnp) * 100, 1)
from 
country