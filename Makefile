eigen_test:	eigen_test.cpp eigen_test.h
	g++ -std=c++11 -g -O0 -I/opt/local/include/eigen3 eigen_test.cpp -I /Users/james/workspace/school/cu/research/arpg/Sophus-master/ -o eigen_test
