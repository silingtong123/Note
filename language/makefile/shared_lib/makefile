SOURCE      :=add.c
OBJS        :=add.o
TARGET      :=libc_s.so

#compile and lib parameter
CC			:=gcc
LIBS        :=
LDFLAGS     :=
DEFINES     :=
INCLUDE     :=-I
CFLAGS      :=
CXXFLAGS    :=

#link
$(TARGET):$(OBJS)
#注意下面的命令需要先使用    Tab       键,再输入命令
	$(CC) -shared $^ -o $@
#注意上面命令，如果有问题换成 $(CC) -shared -o -fpic -o $^ $@

#complie
$(OBJS):$(SOURCE)
	$(CC) -c -fpic $^ -o $@
#clear

#	rm -fr *.o

# 链接静态库举例：g++ -std=c++11 main.cc -L./ -lhello -o main
# export LD_LIBRARY_PATH=/Newdeeprec/gitdownload/test/makefile/:$LD_LIBRARY_PATH