# 第二轮：A 账号空闲卡上的诊断与单景计时（后台），日志在 ~/s2lab/logs/eval2_*.log
$env:PYTHONIOENCODING = 'utf-8'
$local = 'C:/Users/Admin/anaconda3/python.exe'
$py = '/home/u2025140002/second_stage_satellite_rl/envs/DQN/bin/python'
$d = '/home/u2025140002/s2lab/data/v1'
$r = '/home/u2025140002/s2lab/runs2'
$jobs = @(
    @(20654, "$py scripts/s2lab/evaluate.py --dataset-dir $d --runs-dir $r/_copied --split confirm --out /home/u2025140002/s2lab/eval2 --run-ids t14-in-delta-s1 t14-in-delta-s2 t14-in-delta-s3 --diag random_start intf_zero intf_shuffle", 'eval2_diag_t14'),
    @(20655, "$py scripts/s2lab/evaluate.py --dataset-dir $d --runs-dir $r --split confirm --out /home/u2025140002/s2lab/eval2 --run-ids avoid-topk-s1 avoid-topk-s2 avoid-topk-s3 avoid-long-s1 avoid-long-s2 avoid-long-s3 main-ppo-s1 main-ppo-s2 main-ppo-s3 --diag random_start intf_zero intf_shuffle", 'eval2_diag_A2'),
    @(20627, "$py scripts/s2lab/evaluate.py --dataset-dir $d --runs-dir $r --split confirm --out /home/u2025140002/s2lab/eval2_timing --timing --rules greedy dispersed --run-ids main-tsac-s1 main-tsaclegacy-s1 extra-legacydelta-s1 main-ppo-s1 main-dqn-s1 main-cnnsac-s1 main-mlpsac-s1 avoid-topk-s1 avoid-rule-s1 power100-s1; $py scripts/s2lab/evaluate.py --dataset-dir $d --runs-dir $r/_copied --split confirm --out /home/u2025140002/s2lab/eval2_timing --timing --run-ids t14-in-delta-s1", 'eval2_timing')
)
foreach ($j in $jobs) {
    $cmd = "nohup bash -c '$($j[1])' > /home/u2025140002/s2lab/logs/$($j[2]).log 2>&1 < /dev/null & echo started $($j[2])"
    & $local scripts/s2lab/remote2.py sh --port $j[0] --cmd $cmd 2>&1 | Where-Object { $_ -notmatch 'Blowfish|Cryptography|class' }
}
