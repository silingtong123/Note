SOURCE      :=division.c
OBJS        :=division.o
TARGET      :=libc_a.a

#compile and lib parameter
CC			:=gcc
LIBS        :=
LDFLAGS     :=-L
DEFINES     :=
INCLUDE     :=-I
CFLAGS      :=
CXXFLAGS    :=

#link parameter
AR          :=ar

#link
$(TARGET):$(OBJS)
#注意下面的命令需要先使用    Tab       键,再输入命令
	$(AR) -r $@ $^ 

#complie
$(OBJS):$(SOURCE)
	$(CC) -c $^ -o $@
#clear
#	rm -fr *.o 加上这行会导致找不到*.o文件而导致ar失败

# 使用静态库举例： g++ -std=c++11 main.cc -L./ -lc_a -o main
# -L指定静态库目录  -lc_a指链接libc_a.a