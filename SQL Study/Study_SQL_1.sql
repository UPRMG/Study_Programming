select country.name as country_name, city.name as city_name, city.population, country.gnp, city.population*country.gnp as mul
from city
join country
on city.countrycode = country.code
order by mul desc

#select * from city
#select * from country

create table try(
	country_name varchar(45) not null,
	city_name varchar(45) null,
	mul float null
)

insert into try select * from (

select country.name as country_name, city.name as city_name, city.population*country.gnp as mul
from city
join country
on city.countrycode = country.code
order by country_name

) abcd


select * from try

select country_name, sum(mul) from try
group by country_name, city_name with rollup


drop table try