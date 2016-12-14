#!/bin/bash
# Check if flask is installed
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
conda_dir="$DIR/miniconda"
conda_exec="$conda_dir/bin/conda"

## Prepare python 3.4 and flask virtual env
#echo "Will delete $conda_dir"
rm -rf $conda_dir
if [ ! -d "$conda_dir" ] 
then
    rm Miniconda3-latest-Linux-x86_64.sh
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $conda_dir
    rm Miniconda3-latest-Linux-x86_64.sh
    $conda_exec config --add channels conda-forge 
    $conda_exec install -y python=3.5 numpy matplotlib scipy json-c pillow urllib3 glob2 protobuf hdf5 gflags glog openblas boost scikit-image opencv || exit $?
    $conda_exec install -y -c cogsci pygame=1.9.2a0 || exit $?
    $conda_exec install -y -c activisiongamescience python-gflags=3.0.5 || exit $?
    $conda_dir/bin/pip install pyprind
fi
. $conda_dir/bin/activate root 
conda clean --all --yes  || exit $?

export PATH=$PATH:$conda_dir/bin
cd caffe_pp2
make clean || exit $?
make superclean || exit $?
make proto || exit $?
make -j 8  || exit $?
make py -j 4 || exit $?
cd ..
