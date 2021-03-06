#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=as16418 # required to send email notifcations - please replace <your_username> with your college login name or email address
#export PATH=/vol/bitbucket/as16418/myvenv/bin/:$PATH
source /vol/bitbucket/as16418/myvenv/bin/activate
#source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
python3 src/visualisingnewsv2.py
uptime