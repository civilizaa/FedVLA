# 推送到 GitHub

本地仓库已经 `git commit` 完成。只差把远端仓库建好并 push。
下面两种方式二选一。**完成后本文件可删。**

---

## 方式一：给我一个 token，我一次性建好并推送（推荐，最快）

1. 打开 GitHub → 右上角头像 → **Settings → Developer settings →
   Personal access tokens → Tokens (classic)**。
2. **Generate new token (classic)**，勾选 scope：**`repo`**（全选 repo 即可）。
3. 有效期设 7 天足够。
4. 生成后把那串 `ghp_...` 发给我。**token 仅用于这一次建库+推送，用完即可在 GitHub 上撤销（Revoke）。**
5. 我执行：安装 gh → `gh auth login --with-token` → `gh repo create fedvla-repo --private` → `git push`。

> 安全说明：token 只在这一次命令里用，不会写进任何文件；建完库请在 GitHub 上 Revoke。

---

## 方式二：你自己 2 点击 + 1 命令（无需给 token）

1. 浏览器打开 https://github.com/new ，仓库名填 **`fedvla-repo`**，
   选 **Private**，**不要**勾选 Initialize with README（保持空仓库），点 Create。
2. 在你**自己电脑的终端**（Git Bash / PowerShell，不是本沙盒）里：
   ```bash
   cd fedvla-repo
   git branch -M main
   git remote add origin https://github.com/<你的用户名>/fedvla-repo.git
   git push -u origin main
   ```
3. 推送时会弹 **Git Credential Manager** 浏览器登录窗口——那就是授权动作。
   登录后密码处粘贴一次 GitHub 个人令牌（同上 `repo` scope）即可。

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
