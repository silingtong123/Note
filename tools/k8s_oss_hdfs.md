# 常用工具命令

## Kubenetes-k8s

### Pod vs Node

- Pod是Kubernetes抽象出来表示一组应用容器(比如Docker、rkt)，还有这些容器共享的资源
Pod是Kubernetes平台上的原子单位。
- 当我们在Kubernetes上创建一个Deployment时，它将创建附带容器的Pods。每个Pod与被调度的Node绑定直到被删除。万一Node故障了，一个新Pod会被调度到其它可用的Node上

### 常用命令
- 集群信息：kubectl cluster-info 
- 查看配置文件：cat /root/.kube/config  
- 获取所有节点信息: kubectl get pods -A 
- 获取指定命名空间的所有节点信息kubectl get pods -A | grep kubeflow-user-example-com 
- 获取pod_name_1的实时日志流：kubectl logs -f pod_name_1 -n kubeflow-user-example-com 
- 获取pod信息：kubectl describe pod pod_name_1 -n kubeflow-user-example-com 
- 删除某个pod：kubectl delete -f mnist_ps_example.yaml 
- 部署某个pod：kubectl apply -f tftest.yaml  
- 进入某个pod的容器内： kubectl --kubeconfig ~/.kube2/config  -n  kubeflow-user-example-com  exec -it  pod_name_1 -c main  bash 

---

## OSS

### 使用举例
```bash
/home/vincent/projects/slt/ossutil64 cp   gitdownload/DeepRec/tensorflow-1.15.5+deeprec2206-cp36-cp36m-linux_x86_64.whl  oss://rec-deeprec/mms_train_dir/test_uid_ssdhash_4162
```

### 常用命令
- 目录挂载：ossfs rec-deeprec /oss -ourl=oss-cn-beijing-internal.aliyuncs.com
- 上传文件夹：ossutil64 cp -r test_ctr_1221_91/ oss://paramter-backend/deeprec_models_backup/test_ctr_1221_91
- 下载文件夹：ossutil64 cp  --jobs 1 -r  oss://paramter-backend/deeprec_models_backup/test_ctr_1221_91 ./test_ctr_1221_91
- 查看大小：ossutil64  du  oss://paramter-backend/deeprec_models_backup/test_ctr_1221_91
- 查看目录：ossutil64  ls  oss://paramter-backend/deeprec_models_backup/test_ctr_1221_91
- 查看目录和子目录：ossutil64  ls -d oss://rec-deeprec

---

## Hadoop

### 常用命令
- 查看目录下文件：hadoop fs -ls hdfs://10.11.11.241:8020/deepray/deepfm_lr0001
- 查看文件大小：hadoop fs -du -h hdfs://10.11.11.241:8020/hour_dur_add_sam_v3/
- 下载文件：hadoop fs -get hdfs://10.11.11.241:8020/hour_dur_add_sam_v3/
- 上传文件：hadoop fs -put ./test.txt hdfs://10.11.11.241:8020/hour_dur_add_sam_v3/