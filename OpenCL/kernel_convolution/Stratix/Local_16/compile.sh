#!bin


cd 3x3/
aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="s10_gh2e2_4Gx2"
cd ../

cd 5x5/
aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="s10_gh2e2_4Gx2"
cd ../

cd 7x7/
aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="s10_gh2e2_4Gx2"
cd ../

cd 9x9/
aoc -v -report -profile -fp-relaxed kernel_convolution.cl -o bin/kernel_convolution.aocx -board="s10_gh2e2_4Gx2"
cd ../




