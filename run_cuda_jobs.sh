if [[ "$#" -ne 3 ]]
then
    echo "Incorrect number of arguments!"
    echo "Syntax: run_cuda_jobs.sh <num_jobs_per_gpu> <gpus> <command>"
    exit 1
fi

num_jobs_per_gpu=$1
gpus=$2
command=$3

for i in $(seq $num_jobs_per_gpu)
do
    for gpu in $gpus
    do
        echo "Run $i in GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu nohup $command &
    done
done