# 推送到 GitHub

> ✅ 远端仓库已建好：**`https://github.com/hp/FedVLA.git`**（你已用方式二推送成功）。
> 本文件保留作为「身份设置 + 日常收工前 push」的参考。

---

## 推送前先把身份改成你自己的（重要，若之前用了占位身份）

当前首次 commit 用的是占位身份 `FedVLA Author <fedvla@example.com>`，提交不会记到你的账号。
在**你自己的终端**执行（全局，一次即可）：

```bash
git config --global user.name  "你的名字"
git config --global user.email "你的github邮箱"   # 用 GitHub 验证过的邮箱，才有贡献绿点
```

若已用占位身份提交过，可补一句让历史也归你（在本地仓库目录下）：
```bash
git commit --amend --reset-author --no-edit
git push --force-with-lease   # 历史改写后需要强推（首次 push 过的话）
```

---

## 推送前先把身份改成你自己的（重要）

当前 commit 用的是占位身份，提交不会记到你的 GitHub 账号。
在**你自己的终端**执行（全局，一次即可）：

```bash
git config --global user.name  "你的名字"
git config --global user.email "你的github邮箱"   # 用 GitHub 验证过的邮箱，才有贡献绿点
```

若已用占位身份提交过，可补一句让历史也归你：
```bash
git commit --amend --reset-author --no-edit
```

---

## 之后每天收工前

```bash
git add -A
git commit -m "一句话说明今天改了啥"
git push
```
