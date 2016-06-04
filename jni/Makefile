CLASS_PATH = ../classes
JAVAHOME ?= /usr/lib/jvm/java-1.8.0-openjdk-amd64
JNIPATH ?= $(JAVAHOME)/include/
JNIPATH_MD ?= $(JAVAHOME)/include/linux

all : libposix.so

# $* matches the target filename without the extension
posix.h : $(CLASS_PATH)/edu/stanford/nlp/sempre/PosixHelper.class
	$(JAVAHOME)/bin/javah -classpath $(CLASS_PATH) -o $@ edu.stanford.nlp.sempre.PosixHelper
posix.o : posix.h

CPPFLAGS = -I $(JNIPATH) -I $(JNIPATH_MD)
CXXFLAGS = --std=c++0x -Wall -O2 -g

libposix_OBJECTS = posix.o

libposix.so : $(libposix_OBJECTS)
	g++ -shared -o $@ $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $<

%.o : %.cpp
	g++ $(CPPFLAGS) $(CXXFLAGS) -fPIC -DPIC -c $< -o $@

clean :
	rm -f posix.h libposix.so $(libposix_OBJECTS)
