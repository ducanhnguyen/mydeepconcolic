ssh -p 22033 anhnd@uet-hpc.remote.hpc.farm

# HPC

1. upload src of project to hpc:
scp -P 22033 -r /Users/ducanhnguyen/Documents/mydeepconcolic/src anhnd@uet-hpc.remote.hpc.farm:/home/anhnd/mydeepconcolic
scp -P 22033 -r /Users/ducanhnguyen/Documents/mydeepconcolic/src/utils/feature_ranker1d.py anhnd@uet-hpc.remote.hpc.farm:/home/anhnd/mydeepconcolic/src/utils
scp -P 22033 -r /Users/ducanhnguyen/Documents/mydeepconcolic/src/untargeted_fgsm.py anhnd@uet-hpc.remote.hpc.farm:/home/anhnd/mydeepconcolic/src
scp -P 22033 -r /Users/ducanhnguyen/Documents/mydeepconcolic/src/utils anhnd@uet-hpc.remote.hpc.farm:/home/anhnd/mydeepconcolic/src

or download:
scp -P 22033 -r anhnd@uet-hpc.remote.hpc.farm:/home/anhnd/mydeepconcolic/result /Users/ducanhnguyen/Documents/mydeepconcolic/result/fgsm


2. Set up python path
export PYTHONPATH=/home/anhnd/mydeepconcolic/:/home/anhnd/mydeepconcolic/src:/home/anhnd/mydeepconcolic/src/example::/home/anhnd/mydeepconcolic/src/sota:/home/anhnd/mydeepconcolic/src/utils:/home/anhnd/mydeepconcolic/saved_models:$PYTHONPATH

3. Run
goto src
"python deepconcolic.py"

# Install Z3 on HPC
https://cmake.org/download/ (tai cmake)
https://titanwolf.org/Network/Articles/Article?AID=8eb7c0db-de33-4f7a-82e2-f7fb4be81214#gsc.tab=0 (cai cmake ko can sudo)
https://stackoverflow.com/questions/45518317/in-source-builds-are-not-allowed-in-cmake (fix loi “In-source builds are not allowed” in cmake)