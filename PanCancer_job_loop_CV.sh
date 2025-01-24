#!bin/bash

datasets=("BrCa1" "BrCa2" "COAD" "HCC" "PDAC" "PRAD" "RC12" "RC18" "RC20")

for i in "${!datasets[@]}"
do
    dataset=${datasets[$i]}
    target=$((i+1))
    echo ${dataset}
    echo ${target}
    bash /data1/reznike/wangs13/UnitedMet/Performance_Benchmarking/scripts/PanCancer_CV_job "$dataset" "$target"
done