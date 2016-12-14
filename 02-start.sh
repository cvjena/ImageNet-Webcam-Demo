#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LD_LIBRARY_PATH=$DIR/miniconda/lib:/home/dbvdemo/lib/cuda7.5/lib64:$LD_LIBRARY_PATH
    
conda_dir="$DIR/miniconda"
if [ ! -d "$conda_dir" ]
then
    echo "Installation script was not execute, please execute install.sh!"
fi
. $conda_dir/bin/activate root

python livedemo.py