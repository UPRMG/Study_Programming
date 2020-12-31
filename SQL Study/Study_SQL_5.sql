-- 나라별 인구 체크
select name, round(population/100000000,1) as popul
from country
order by popul desc

-- 사용 언어별 인구 체크
select countrylanguage.language, sum(countrylanguage.percentage*country.population) as popul
from country
join countrylanguage
on country.code = countrylanguage.countrycode
group by language
order by popul desc




