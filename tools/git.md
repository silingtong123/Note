#### 建立新仓库
```
git init  本地初始化
远程创建同名仓库
git remote add origin git@github.com:silingtong123/Note.git 本地添加远程仓库
git add .
git commit -m "..."
git push origin master 本地分支推送到远端
```

#### ssh秘钥生成
```
git config --list
git config --global user.name "lingtong.si"
git config --global user.email "lingtong.si@appshahe.com"
apt-get install ssh-client #安装ssh客户端 
ssh-keygen -t rsa -C "lingtong.si@appshahe.com" 

# 在github上添加新生成的ssh秘钥
```
- 查看公钥：vim ~/.ssh/authorized_keys
#### 常见问题
- unable to access '.git/': Failed to connect to github.com port 443: Connection timed out
  - 取消代理：
    - git config --global --unset http.proxy
    - git config --global --unset https.proxy
  - 添加代理：
    - git config --global  http.proxy http://10.0.24.95:8888
    - export http_proxy=http://10.0.24.95:8888
    - export https_proxy=http://10.0.24.95:8888

- OpenSSL SSL_read: Connection was reset, errno 10054 SSL证书没有经过第三方机构的签署 ， 取消SSL证书验证
  - git config --global http.sslVerify "false"
  - git config --global https.sslVerify "false"

- error: invalid path"" 某分支下的文件名格式不支持，最终导致在git clone的时候找不到这个文件路径导致的
  - git config core.protectNTFS false