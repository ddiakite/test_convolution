date
hostname
source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t S10OAPI
cd /home/u79811/ONEAPI/con2D_ONEAPI/Naive/3x3
make clean -f Makefile.fpga
make hw -f Makefile.fpga
make run_hw -f Makefile.fpga
date
