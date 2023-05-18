#导入模块
import multiprocessing
import time
 
#定义进程执行函数
def clock(interval, name=None):
    print(name)
    for i in range(5):
        print('当前时间为{0}： '.format(time.ctime()))
        time.sleep(interval)
 
if __name__=='__main__':
	#创建子进程
	p=multiprocessing.Process(target=clock,args=(1,"test_name"))
	#启动子进程
	p.start()
	p.join()
	#获取进程的 ID
	print('p.id:',p.pid)
	#获取进程的名称
	print('p.name:',p.name)
	#判断进程是否运行
	print('p.is_alive:',p.is_alive())