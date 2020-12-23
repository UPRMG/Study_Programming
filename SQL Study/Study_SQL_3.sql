select num1, count(a.num1)/(
#1-select : 확률 분모값 (전체 나온 횟수)
select count(num1)
from
(select seq_no, draw_date, num1
from lotto_master
union all
select seq_no, draw_date, num2
from lotto_master
union all
select seq_no, draw_date, num3
from lotto_master
union all
select seq_no, draw_date, num4
from lotto_master
union all
select seq_no, draw_date, num5
from lotto_master
union all
select seq_no, draw_date, num6
from lotto_master) t
#1-select : 확률 분모값 (전체 나온 횟수)
)*100 as lr,

#select : 전체 평균값 
(
select avg(gtr.lr)
from
(
#2-select : gtr.lr을 위한 number별 평균
select num1, count(a.num1)/
(
select count(num1)
from
(select seq_no, draw_date, num1
from lotto_master
union all
select seq_no, draw_date, num2
from lotto_master
union all
select seq_no, draw_date, num3
from lotto_master
union all
select seq_no, draw_date, num4
from lotto_master
union all
select seq_no, draw_date, num5
from lotto_master
union all
select seq_no, draw_date, num6
from lotto_master) tt
)*100 as lr
from 
(
select seq_no, draw_date, num1
from lotto_master
union all
select seq_no, draw_date, num2
from lotto_master
union all
select seq_no, draw_date, num3
from lotto_master
union all
select seq_no, draw_date, num4
from lotto_master
union all
select seq_no, draw_date, num5
from lotto_master
union all
select seq_no, draw_date, num6
from lotto_master
) as a
group by a.num1
#2-select
) gtr
) as avge

#1-from : number union all
from 
(
select seq_no, draw_date, num1
from lotto_master
union all
select seq_no, draw_date, num2
from lotto_master
union all
select seq_no, draw_date, num3
from lotto_master
union all
select seq_no, draw_date, num4
from lotto_master
union all
select seq_no, draw_date, num5
from lotto_master
union all
select seq_no, draw_date, num6
from lotto_master
) as a
#1-from
group by a.num1
order by lr desc



