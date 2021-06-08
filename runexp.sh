set -e

datadir=dataset
dataurl="https://ndownloader.figshare.com/files/27568763"
dataname="embedded.zip"
if [ -f "$dataname" ] ; then
    echo "dataset '$dataname' already present, skipping download"
else 
    echo "downloading dataset to '$datadir' folder"
    wget -O $dataname $dataurl
fi
if [ -d "$datadir" ] ; then
    echo " '$datadir' folder already present, skipping unpaking"
else 
    echo "unpackiong '$dataname' to '$datadir' folder"
    mkdir -p dataset
    unzip -q $dataname   -d $datadir
    mv ${datadir}/hierarchy_frames/* ${datadir}/
    rmdir ${datadir}/hierarchy_frames/
fi

PYTHONPATH=. python  scripts/fs2desc.py dataset descriptor.json
nexp=$(ls -1 inputs | wc -l) 
echo "loading $nexp experiments" 
counter=1 
mkdir -p results
 for i in inputs/* ; do 
     o=results/$(basename $i).npy.lz4
     if [ -f "$o" ] ; then
        echo [${counter}/${nexp}]: ${i} already done, skipping 
    else 
        echo -n [${counter}/${nexp}]": "
        PYTHONPATH=. python scripts/json_train.py  --results ${o} ${i} 
    fi 
    : $((counter++)) 
 done 
 echo creating figures... 

tmpf=$(mktemp -d)
echo -n [1/3]": "
PYTHONPATH=. python  scripts/plot_hierarchy.py results/o_b_1.json.npy.lz4 results/a{95,90,80}_b_1.json.npy.lz4 --labels "full sup,a=0.95,a=0.90,a=0.80" --o ${tmpf}/semi >/dev/null
echo "done!"
echo -n [2/3]": "
PYTHONPATH=. python  scripts/plot_hierarchy.py  results/dummy.json.npy.lz4 results/o_b_1.json.npy.lz4 --labels "encounter, predict endounter" -o ${tmpf}/full >/dev/null
echo "done!"
echo -n [3/3]": "
PYTHONPATH=. python scripts/plot_hierarchy.py results/{a95_d,a90_b}_1.json.npy.lz4 --labels "devel,random"  -o ${tmpf}/setting >/dev/null
echo "done!"

mkdir -p outputs
mv  ${tmpf}/{fullcost,semihf,semisup,settinghf,settingsup}.png outputs/
rm -r ${tmpf}

