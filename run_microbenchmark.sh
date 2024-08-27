./microbenchmark -m L1_SIZE -s REG_128BIT -b ALL -i 3400000 > L1_128bit_benchmark.txt
echo "DONE L1_128bit_benchmark.txt"
./microbenchmark -m L2_SIZE -s REG_128BIT -b ALL -i 40000 > L2_128bit_benchmark.txt
echo "DONE L2_128bit_benchmark.txt"
./microbenchmark -m L3_SIZE -s REG_128BIT -b ALL -i 2500 > L3_128bit_benchmark.txt
echo "DONE L3_128bit_benchmark.txt"
./microbenchmark -m MM_SIZE -s REG_128BIT -b ALL -i 155 > MM_128bit_benchmark.txt
echo "DONE MM_128bit_benchmark.txt"


./microbenchmark -m L1_SIZE -s REG_256BIT -b ALL -i 3400000 > L1_256bit_benchmark.txt
echo "DONE L1_256bit_benchmark.txt"
./microbenchmark -m L2_SIZE -s REG_256BIT -b ALL -i 40000 > L2_256bit_benchmark.txt
echo "DONE L2_256bit_benchmark.txt"
./microbenchmark -m L3_SIZE -s REG_256BIT -b ALL -i 2500 > L3_256bit_benchmark.txt
echo "DONE L3_256bit_benchmark.txt"
./microbenchmark -m MM_SIZE -s REG_256BIT -b ALL -i 155 > MM_256bit_benchmark.txt
echo "DONE MM_256bit_benchmark.txt"
