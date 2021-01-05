
-- gnp 평균보다 높고 낮은 기준을 구분하고, 과거 gnp대비 성장률 기준으로 내림차순 정렬 후 기대수명 70 이상의 국가 추출
select code, 
	   name, 
	   continent, 
	   population, 
	   gnp, 

case when gnp > (select avg(gnp) from country) then 'upper_avg'
	 when gnp < (select avg(gnp) from country) then 'lower_avg'
	 end avg,
	 
	 IFNULL((gnp-gnpold)/gnpold*100, 0) as rate,
	 lifeexpectancy

from country
where lifeexpectancy > 70
order by rate desc


