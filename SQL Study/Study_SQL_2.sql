use world

select *
from city

select *
from country

select *
from countrylanguage

#나라별 gnp순 상승률 및 면적대비 인구 현황 등 자료 추출
select b.name, b.continent, b.region, b.surfacearea, b.population, b.gnpold, b.gnp, b.high_rate, round(b.rate_surface,1) as rate_s, count(a.language) as C_lan, sum(a.percentage) as lan_per
from countrylanguage as a,
(select code, continent, region, name, surfacearea, population, gnpold, gnp, round((gnp-gnpold)/gnpold*100,1) as high_rate, population/surfacearea as rate_surface
from country
where population >= 40000000 and gnp >= 300000) as b
where a.countrycode = b.code
group by a.countrycode
order by b.gnp desc

#도시별 언어 비율 대비 실제 사용 인구 추출
select city.countrycode, countrylanguage.language, city.population * countrylanguage.percentage as popul
from city, countrylanguage
where city.countrycode = countrylanguage.countrycode
order by countrylanguage.language

#나라별 도시 개수와 인구 추출
select countrycode, count(NAME) as C_name, sum(population) as population
from city
group by countrycode 
order by C_name desc