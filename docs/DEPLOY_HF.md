# 部署到 Hugging Face Spaces（免费 · 24h 在线 · 永久 URL）

> **目标**：让任何人通过 `https://huggingface.co/spaces/<你的用户名>/<space-名>` 直接访问，不需要你的电脑开机。
>
> 全程预计 **10～15 分钟**。

---

## 0. 准备

- 一个 Hugging Face 账号（免费注册：https://huggingface.co/join）
- 本机已装 `git`（`git --version` 可验证）
- 本项目代码已在你电脑上（即 `D:\true\ecommerce_ai_project`）

---

## 1. 在 HF 上创建 Space

1. 登录 https://huggingface.co/
2. 右上角头像 → **New Space**
3. 填写：
   - **Space name**：`ecommerce-ai`（或任何你喜欢的名字）
   - **License**：MIT
   - **Select the Space SDK**：选 **Gradio**
   - **Space hardware**：**CPU basic - 2 vCPU · 16GB · FREE**
   - **Visibility**：Public（朋友才能直接打开）
4. 点 **Create Space**

创建后会给你一个 git 仓库地址，形如：

```
https://huggingface.co/spaces/<你的用户名>/ecommerce-ai
```

---

## 2. 配置 Secrets（密钥）

在你刚创建的 Space 页面：

1. 点右上角 **Settings**
2. 找到 **Variables and secrets** → **New secret**
3. 添加这两个（**Name 必须完全一致**）：

   | Name | Value | 必填 |
   |------|-------|------|
   | `ADMIN_PASSWORD` | 你的商家后台密码（自己定） | ✅ 必填，否则无法登录后台 |
   | `DEEPSEEK_API_KEY` | DeepSeek API Key（`sk-...`） | ⚪ 可选，没有的话 AI 文案降级为模板 |

   > **不要把这两个写在代码里！** 写在这里 HF 会加密存储。

---

## 3. 把代码推到 Space

在项目目录下打开 PowerShell：

```powershell
cd D:\true\ecommerce_ai_project

# 如果还没初始化过 git
git init
git add .
git commit -m "Initial commit: multimodal AI ecommerce"

# 关联 HF Space 仓库（替换成你自己的）
git remote add space https://huggingface.co/spaces/<你的用户名>/ecommerce-ai

# 推送（首次会让你输入 HF 用户名 + 访问令牌）
git push space HEAD:main
```

### 关于 HF 访问令牌

第一次 `git push` 时会要求登录：

- **用户名**：你的 HF 用户名
- **密码**：**不是网页密码**！要去 https://huggingface.co/settings/tokens 生成一个 **Access Token**（权限选 `write`），把那串 `hf_xxxxxx` 当密码粘进去。

> 推荐用 [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/quick-start#login) 登录省事：
> ```powershell
> pip install huggingface_hub
> huggingface-cli login
> ```
> 之后 `git push` 不再要密码。

---

## 4. 等待构建（约 5～10 分钟）

回到 Space 页面，会看到顶部状态条从 **Building** → **Running**。

构建过程在右上角 **Logs** 里能看到实时输出，类似：

```
==========
Building...
Installing requirements: gradio, torch, transformers, faiss-cpu...
==========
Loading CLIP model...
CLIP loaded.
Loaded 201 visible products for AI index.
Encoding product images...
FAISS index built.
* Running on local URL: http://0.0.0.0:7860
```

**首次启动较慢**（要下 CLIP 模型 + 给 201 张图算 embedding），约 3～5 分钟。

---

## 5. 完成

页面会自动嵌入你的 Gradio 界面。把这个 URL 发给任何人：

```
https://huggingface.co/spaces/<你的用户名>/ecommerce-ai
```

也支持直链：

```
https://<你的用户名>-ecommerce-ai.hf.space
```

---

## 后续：改了代码怎么更新？

```powershell
git add .
git commit -m "feat: 加了新功能"
git push space HEAD:main
```

HF Space 检测到 push 会自动重建（约 2~5 分钟）。

---

## 常见问题

### Q: 数据会保存吗？

免费 CPU Space 是**临时文件系统**：
- ✅ `data/products.csv`、`data/images/` 这些**仓库里的**文件每次重启都还在
- ❌ 顾客注册账号 `data/users.json`、订单 `data/orders.json` 在 Space **重启后会重置**

对于面试演示完全够用。如果想数据持久化：
- 升级 HF Persistent Storage（$5/月）
- 或改用 Supabase / Hugging Face Datasets 存数据

### Q: 我的 Space 一直在 Building 卡住？

看 **Logs** 找报错。常见原因：
- `requirements.txt` 有装不上的版本 → 把版本号宽松一点
- 内存超 16GB → 换更小的模型或别一次性建 1000+ 商品的索引

### Q: 怎么不让别人乱看？

Space Settings → **Visibility** 改成 **Private**。但 Private Space 只能你自己看，无法分享。

折中：Public 但商家后台用强密码（`ADMIN_PASSWORD`）。

### Q: 想要永久网址 / 自己域名？

HF Space 自带的 `*.hf.space` 已经是永久网址了。  
想要 `shop.yourname.com`：付费 Space + DNS CNAME，或者自己 VPS。

### Q: 怎么看访问量？

Space Settings → **Analytics**（需要 Public Space）。

---

## 部署完成 Checklist

- [ ] Space 状态 `Running`
- [ ] 首页能看到商品商城
- [ ] 注册一个测试账号 → 加购 → 下单 → 进入「我的订单」
- [ ] 用 `ADMIN_PASSWORD` 登录商家后台 → 「订单管理」能看到刚下的单
- [ ] 把 URL 发给一个朋友 → 朋友打开不报错

把 URL 加到你简历的项目栏，比"GitHub 链接"更直接、更有说服力。
