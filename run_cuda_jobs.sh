if [[ "$#" -ne 3 ]]
then
    echo "Incorrect number of arguments!"
    echo "Syntax: run_cuda_jobs.sh <num_jobs_per_gpu> <num_gpus> <command>"
    exit 1
fi

num_jobs_per_gpu=$1
num_gpus=$2
command=$3

for i in $(seq $num_jobs_per_gpu)
do
    for gpu in $(seq $num_gpus)
    do
        echo "CUDA_VISIBLE_DEVICES=$(( $gpu - 1)) $command &"
        CUDA_VISIBLE_DEVICES=$(( $gpu - 1)) $command &
    done
done