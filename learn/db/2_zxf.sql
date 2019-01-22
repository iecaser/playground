select
	c.user_log_acct,
	c.batch_id,
	sum(
		case c.cps_state_cd
			when 2
			then 1
			else 0
		end) as giveup_num
from
	(
		select
			a.*
		from
			(
				select distinct batch_id from tmp.zxf_user_use_batch
			)
			b
		join
			(
				select
					user_log_acct,
					batch_id,
					cps_state_cd
				from
					gdm.gdm_m07_cps_basic_info
				where
					dt = '2018-12-01'
			)
			a
		on
			a.batch_id = b.batch_id
	)
	c
group by
	c.user_log_acct,
	c.batch_id