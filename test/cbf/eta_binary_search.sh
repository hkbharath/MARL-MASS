# collision scenario 1: HSS with CBF-AV state in longitudinal crash condition
python -m test.cbf.test_vehicle_safety --test-type lon --safety none --extreme --exp-name eta_exp-tau:0.5 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lon --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.5 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lon --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.25 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lon --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.125 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lon --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.0625 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lon --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.03125 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lon --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.046875 > results/cav-test.log

collision scenario 2: HSS with CBF-AV state for in crash condition while changing lane
python -m test.cbf.test_vehicle_safety --test-type lat_adj_left_lc --safety none --extreme --exp-name eta_exp-tau:0.5 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lat_adj_left_lc --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.5 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lat_adj_left_lc --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.25 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lat_adj_left_lc --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.125 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lat_adj_left_lc --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.0625 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lat_adj_left_lc --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.03125 > results/cav-test.log
python -m test.cbf.test_vehicle_safety --test-type lat_adj_left_lc --safety cbf-avs_cint --extreme --exp-name eta_exp-tau:0.5 --gamma 0.046875 > results/cav-test.log