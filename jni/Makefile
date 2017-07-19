CLASS_PATH = ../classes
JAVAHOME ?= /usr/lib/jvm/java-1.8.0
JNIPATH ?= $(JAVAHOME)/include/
JNIPATH_MD ?= $(JAVAHOME)/include/linux

all : libposix.so libhunspell_jni.so

# $* matches the target filename without the extension
posix.h : $(CLASS_PATH)/edu/stanford/nlp/sempre/PosixHelper.class
	$(JAVAHOME)/bin/javah -classpath $(CLASS_PATH) -o $@ edu.stanford.nlp.sempre.PosixHelper

CPPFLAGS := -I $(JNIPATH) -I $(JNIPATH_MD)
CXXFLAGS := --std=c++14 -Wall -O2 -g -fPIC -DPIC

libposix.so : posix.h posix.cpp
	g++ -shared -o $@ $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -fPIC -DPIC $^

hunspell-jni.h : $(CLASS_PATH)/corenlp/edu/stanford/nlp/sempre/corenlp/HunspellDictionary.class
	$(JAVAHOME)/bin/javah -classpath $(CLASS_PATH):$(CLASS_PATH)/corenlp -o $@ edu.stanford.nlp.sempre.corenlp.HunspellDictionary

libhunspell_jni_OBJECTS = hunspell-jni.o
libhunspell_jni_CPPFLAGS := $(CPPFLAGS) $(shell pkg-config --cflags hunspell)
libhunspell_jni_LIBS := $(shell pkg-config --libs hunspell)

libhunspell_jni.so : hunspell-jni.h hunspell-jni.cpp
	g++ -shared -o $@ $(libhunspell_jni_CPPFLAGS) $(CXXFLAGS) $(libhunspell_jni_LIBS) $^

clean :
	rm -f posix.h libposix.so hunspell-jni.h libhunspell-jni.so
