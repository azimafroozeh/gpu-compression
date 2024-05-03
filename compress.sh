bin=./bin/ssb/binpack
rbin=./bin/ssb/rlebinpack
dbin=./bin/ssb/deltabinpack

arr=("lo_custkey" "lo_partkey" "lo_suppkey" "lo_orderdate" "lo_quantity" "lo_extendedprice" "lo_discount" "lo_revenue" "lo_supplycost" "lo_orderkey" "lo_linenumber" "lo_tax" "lo_ordtotalprice" "lo_commitdate")
for val in ${arr[*]}; do
 echo $val
 $bin $val
 $dbin $val
 $rbin $val
done
