-- 사용 언어의 비율이 70% 이상의 언어 종류별 개수 체크
select language, count(language) as count
from countrylanguage
where percentage > 70
group by language
order by count desc

-- NULL Value 변경
select name, IFNULL(indepyear, 0) as basic_test
from country

-- case when basic
select 
case when surfacearea > 50000 then 'high 50000'
end 
from country
