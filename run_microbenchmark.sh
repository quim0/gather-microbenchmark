./microbenchmark -m L1_SIZE -s REG_128BIT -b ALL -i 7500000
echo "DONE L1_128bit_benchmark.txt\n"
./microbenchmark -m L2_SIZE -s REG_128BIT -b ALL -i 75000
echo "DONE L2_128bit_benchmark.txt\n"
./microbenchmark -m L3_SIZE -s REG_128BIT -b ALL -i 3000
echo "DONE L3_128bit_benchmark.txt\n"
./microbenchmark -m MM_SIZE -s REG_128BIT -b ALL -i 150 
echo "DONE MM_128bit_benchmark.txt\n"


./microbenchmark -m L1_SIZE -s REG_256BIT -b ALL -i 7500000s
echo "DONE L1_256bit_benchmark.txt\n"
./microbenchmark -m L2_SIZE -s REG_256BIT -b ALL -i 75000 
echo "DONE L2_256bit_benchmark.txt\n"
./microbenchmark -m L3_SIZE -s REG_256BIT -b ALL -i 3000   
echo "DONE L3_256bit_benchmark.txt\n"
./microbenchmark -m MM_SIZE -s REG_256BIT -b ALL -i 150 
echo "DONE MM_256bit_benchmark.txt\n"

./microbenchmark -m L1_SIZE -s REG_512BIT -b ALL -i 7500000
echo "DONE L1_512bit_benchmark.txt\n"
./microbenchmark -m L2_SIZE -s REG_512BIT -b ALL -i 75000
echo "DONE L2_512bit_benchmark.txt\n"
./microbenchmark -m L3_SIZE -s REG_512BIT -b ALL -i 3000
echo "DONE L3_512bit_benchmark.txt\n"
./microbenchmark -m MM_SIZE -s REG_512BIT -b ALL -i 150
echo "DONE MM_512bit_benchmark.txt\n"
