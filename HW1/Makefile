CXXFLAGS = $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

CXXC = clang++

SOURCES = $(wildcard *.cpp)

main : main.cpp Examples.cpp
	$(CXXC) $(CXXFLAGS) --std=c++23 $(LIBS) -o $@ $^

hw : hw1.cpp
	$(CXXC) $(CXXFLAGS) --std=c++23 $(LIBS) -o $@ $^


