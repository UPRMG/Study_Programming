-- 인구 5천만 이상 국가의 사용 언어 개수 파악
select name, count(name)
from countrylanguage as a, 
(select code, name
from country
where population > 50000000) as b
where a.countrycode = b.code
group by b.name

