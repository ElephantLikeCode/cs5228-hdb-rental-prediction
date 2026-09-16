# CS5228 Project 本地环境

## 推荐配置

- Python：`3.12`
- 虚拟环境：项目根目录下的 `.venv`
- Jupyter 内核：`CS5228 Project (Python 3.12)`

虚拟环境不应提交或从其他电脑复制。每位组员应通过 `requirements.txt`
在自己的电脑上重建环境。

## 日常使用

在 PowerShell 中运行：

```powershell
cd '<你的项目目录>'
& '.\.venv\Scripts\Activate.ps1'
python -m pytest -q
```

启动 JupyterLab：

```powershell
jupyter lab
```

打开 notebook 后选择 `CS5228 Project (Python 3.12)` 内核。

退出虚拟环境：

```powershell
deactivate
```

## 从依赖清单重建

```powershell
cd '<你的项目目录>'
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m ipykernel install --user --name cs5228-project --display-name 'CS5228 Project (Python 3.12)'
```

macOS/Linux 用户将激活命令替换为：

```bash
source .venv/bin/activate
```
