CC=gcc
SOURCES=gathermicrobench/microbenchmark.c gathermicrobench/generate_data.c
CFLAGS=-Wall -Wextra -I. -march=native

all:
	$(CC) $(CFLAGS) -O3 $(SOURCES) -o gather_microbench

debug:
	$(CC) $(CFLAGS) -g $(SOURCES) -o gather_microbench

clean:
	rm -f gather_microbench