#!bin
source /opt/intelFPGA_pro/18.1/hld/flik_init.sh 

#cd 3x3/
#aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="flik"
#cd ../

#cd 5x5/
#aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="flik"
#cd ../

cd 7x7/
aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="flik"
cd ../

#cd 9x9/
#aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="flik"
#cd ../

cd 15x15/
aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="flik"
cd ../
