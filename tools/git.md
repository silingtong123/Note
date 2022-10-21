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
ssh-keygen -t rsa -C "lingtong.si@appshahe.com" 

# 在github上添加新生成的ssh秘钥
```