mkdir results
./bin/gathermicrobench -m L1_SIZE -s REG_128BIT -b ALL -i 3400000 > results/L1_128bit_benchmark.txt
echo "DONE L1_128bit_benchmark.txt"
./bin/gathermicrobench -m L2_SIZE -s REG_128BIT -b ALL -i 40000 > results/L2_128bit_benchmark.txt
echo "DONE L2_128bit_benchmark.txt"
./bin/gathermicrobench -m L3_SIZE -s REG_128BIT -b ALL -i 3750 > results/L3_128bit_benchmark.txt
echo "DONE L3_128bit_benchmark.txt"
./bin/gathermicrobench -m MM_SIZE -s REG_128BIT -b ALL -i 155 > results/MM_128bit_benchmark.txt
echo "DONE MM_128bit_benchmark.txt"

./bin/gathermicrobench -m L1_SIZE -s REG_128BIT -b STRIDE --stride 2 -i 3400000 > results/L1_128bit_stride2.txt
echo "DONE L1_128bit STRIDE=2"
./bin/gathermicrobench -m L2_SIZE -s REG_128BIT -b STRIDE --stride 2 -i 40000 > results/L2_128bit_stride2.txt
echo "DONE L2_128bit STRIDE=2"
./bin/gathermicrobench -m L3_SIZE -s REG_128BIT -b STRIDE --stride 2 -i 3750 > results/L3_128bit_stride2.txt
echo "DONE L3_128bit STRIDE=2"
./bin/gathermicrobench -m MM_SIZE -s REG_128BIT -b STRIDE --stride 2 -i 155 > results/MM_128bit_stride2.txt
echo "DONE MM_128bit STRIDE=2"

./bin/gathermicrobench -m L1_SIZE -s REG_128BIT -b STRIDE --stride 4 -i 3400000 > results/L1_128bit_stride4.txt
echo "DONE L1_128bit STRIDE=4"
./bin/gathermicrobench -m L2_SIZE -s REG_128BIT -b STRIDE --stride 4 -i 40000 > results/L2_128bit_stride4.txt
echo "DONE L2_128bit STRIDE=4"
./bin/gathermicrobench -m L3_SIZE -s REG_128BIT -b STRIDE --stride 4 -i 3750 > results/L3_128bit_stride4.txt
echo "DONE L3_128bit STRIDE=4"
./bin/gathermicrobench -m MM_SIZE -s REG_128BIT -b STRIDE --stride 4 -i 155 > results/MM_128bit_stride4.txt
echo "DONE MM_128bit STRIDE=4"

./bin/gathermicrobench -m L1_SIZE -s REG_256BIT -b ALL -i 3400000 > results/L1_256bit_benchmark.txt
echo "DONE L1_256bit_benchmark.txt"
./bin/gathermicrobench -m L2_SIZE -s REG_256BIT -b ALL -i 40000 > results/L2_256bit_benchmark.txt
echo "DONE L2_256bit_benchmark.txt"
./bin/gathermicrobench -m L3_SIZE -s REG_256BIT -b ALL -i 3750 > results/L3_256bit_benchmark.txt
echo "DONE L3_256bit_benchmark.txt"
./bin/gathermicrobench -m MM_SIZE -s REG_256BIT -b ALL -i 155 > results/MM_256bit_benchmark.txt
echo "DONE MM_256bit_benchmark.txt"

./bin/gathermicrobench -m L1_SIZE -s REG_256BIT -b STRIDE --stride 2 -i 3400000 > results/L1_256bit_stride2.txt
echo "DONE L1_256bit STRIDE=2"
./bin/gathermicrobench -m L2_SIZE -s REG_256BIT -b STRIDE --stride 2 -i 40000 > results/L2_256bit_stride2.txt
echo "DONE L2_256bit STRIDE=2"
./bin/gathermicrobench -m L3_SIZE -s REG_256BIT -b STRIDE --stride 2 -i 3750 > results/L3_256bit_stride2.txt
echo "DONE L3_256bit STRIDE=2"
./bin/gathermicrobench -m MM_SIZE -s REG_256BIT -b STRIDE --stride 2 -i 155 > results/MM_256bit_stride2.txt
echo "DONE MM_256bit STRIDE=2"

./bin/gathermicrobench -m L1_SIZE -s REG_256BIT -b STRIDE --stride 4 -i 3400000 > results/L1_256bit_stride4.txt
echo "DONE L1_256bit STRIDE=4"
./bin/gathermicrobench -m L2_SIZE -s REG_256BIT -b STRIDE --stride 4 -i 40000 > results/L2_256bit_stride4.txt
echo "DONE L2_256bit STRIDE=4"
./bin/gathermicrobench -m L3_SIZE -s REG_256BIT -b STRIDE --stride 4 -i 3750 > results/L3_256bit_stride4.txt
echo "DONE L3_256bit STRIDE=4"
./bin/gathermicrobench -m MM_SIZE -s REG_256BIT -b STRIDE --stride 4 -i 155 > results/MM_256bit_stride4.txt
echo "DONE MM_256bit STRIDE=4"
