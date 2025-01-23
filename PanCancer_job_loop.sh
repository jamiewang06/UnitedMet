#!bin/bash

datasets=("BrCa1" "BrCa2" "COAD" "HCC" "PDAC" "PRAD" "RC12" "RC18" "RC20")

for dataset in "${datasets[@]}"
do
    bash /data1/reznike/wangs13/UnitedMet/Performance_Benchmarking/scripts/PanCancer_job "$dataset"
done