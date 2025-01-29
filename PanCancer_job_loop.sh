#!bin/bash

datasets=("BrCa1" "BrCa2" "COAD" "HCC" "PDAC" "PRAD" "RC12" "RC18" "RC20")
ndims_list=(41 41 51 51 41 51 31 41 51)


for i in "${!datasets[@]}"
do
    dataset=${datasets[$i]}
    ndims=${ndims_list[$i]}
    echo ${dataset}
    echo ${ndims}

    bash /data1/reznike/wangs13/UnitedMet/Performance_Benchmarking/scripts/PanCancer_job "$dataset" "$ndims"
done