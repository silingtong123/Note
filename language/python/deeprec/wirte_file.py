import os

for rank in range(5):
    fd = os.open("prefix"+str(rank)+"_SUCCESS",os.O_RDWR|os.O_CREAT)
    os.close(fd)