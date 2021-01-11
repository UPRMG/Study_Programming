-- 공식 언어 50% 이상 사용 언어 정리
select *
from countrylanguage
where isofficial = 'T' and percentage > 50



-- 공식 언어 50% 이상 사용 언어 코드에 따른 country 조인
select *
from 
(select *
from countrylanguage
where isofficial = 'T' and percentage > 50) as a
join country 
on a.countrycode = country.code


