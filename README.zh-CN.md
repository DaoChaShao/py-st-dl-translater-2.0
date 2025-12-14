<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**项目简介**
---

本项目是一个**机器翻译实践学习项目**，从基础的**RNN序列到序列模型**（1.0版）升级至**带注意力的增强架构**（2.0版），全程使用*
*PyTorch**实现。2.0版本新增**注意力机制**
，让解码器能够动态关注源语句的关键部分，显著提升翻译质量，特别是对长句的处理能力。系统使用[AI Challenger 2017](https://tianchi.aliyun.com/dataset/174937)
中英平行语料库进行训练与评估，帮助深入理解从传统到现代的序列翻译技术。


**数据描述**
---

+ 基本信息
    - 来源：天池（阿里云）AI Studio
    - 数据集ID：174937
    - 标题：英中翻译数据集（AI Studio）
    - 格式：文本文件（UTF-8编码）

+ 数据结构
    - 数据集包含平行句对的英文和中文：
    - 英文句子：自然英文文本
    - 中文句子：对应的简体中文翻译

+ 统计信息
    - 总句对数量：1,200,000+ (120万)
    - 英文句子平均长度：15-25个单词
    - 中文句子平均长度：10-20个字符
    - 英文词汇量：约50,000-100,000词
    - 中文字符量：约5,000-8,000字

**快速开始**
---

1. 将本仓库克隆到本地计算机。
2. 使用以下命令安装所需依赖项：`pip install -r requirements.txt`
3. 使用以下命令运行应用程序：`streamlit run main.py`
4. 你也可以通过点击以下链接在线体验该应用：  
   [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://rnn-translater-2.streamlit.app/)

**隐私声明**
---
本应用程序旨在处理您提供的数据以生成定制化的建议和结果。您的隐私至关重要。

**我们不会收集、存储或传输您的个人信息或数据。** 所有处理都在您的设备本地进行（在浏览器或运行时环境中），*
*数据永远不会发送到任何外部服务器或第三方。**

- **本地处理：** 您的数据永远不会离开您的设备。整个分析和生成过程都在本地进行。
- **无数据保留：** 由于没有数据传输，因此不会在任何服务器上存储数据。关闭应用程序通常会清除任何临时本地数据。
- **透明度：** 整个代码库都是开源的。我们鼓励您随时审查[代码](./)以验证您的数据处理方式。

总之，您始终完全控制和拥有自己的数据。

**许可声明**
---
本项目是开源的，可在 [BSD-3-Clause 许可证](LICENCE) 下使用。

简单来说，这是一个非常宽松的许可证，允许您几乎出于任何目的自由使用此代码，包括在专有项目中，只要您包含原始的版权和许可证声明。

欢迎随意分叉、修改并在此作品基础上进行构建！我们只要求您在适当的地方给予认可。

**环境设置**
---
本项目使用 **Python 3.12** 和 [uv](https://docs.astral.sh/uv/) 进行快速的依赖管理和虚拟环境处理。所需的 Python
版本会自动从 [.python-version](.python-version) 文件中检测到。

1. **安装 uv**：  
   如果您还没有安装 `uv`，可以使用以下命令安装：
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 此安装方法适用于 macOS 和 Linux。
    ```
   或者，您可以运行以下 PowerShell 命令来安装：
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # 此安装方法适用于 Windows。
    ```

   **💡 推荐**：为了获得最佳体验，请将 `uv` 作为独立工具安装。避免在 `pip` 或 `conda` 环境中安装，以防止潜在的依赖冲突。

2. **添加依赖**：

- 添加主要（生产）依赖：
    ```bash
    uv add <package_name>
    # 这会自动更新 pyproject.toml 并安装包
    ```
- 添加开发依赖：
    ```bash
    uv add <package_name> --group dev
    # 示例：uv add ruff --group dev
    # 这会自动将包添加到 [project.optional-dependencies.dev] 部分
    ```
- 添加其他类型的可选依赖（例如测试、文档）：
    ```bash
    uv add <package_name> --group test
    uv add <package_name> --group docs
    ```
- 从 `requirements.txt` 文件导入依赖：
    ```bash
    uv add -r requirements.txt
    # 这会从 requirements.txt 读取包并将其添加到 pyproject.toml
    ```
- 从当前依赖生成 `requirements.txt` 文件：
    ```bash
    # 这会将所有依赖（包括可选依赖）导出到 requirements-all.txt
    uv pip compile pyproject.toml --all-extras -o requirements.txt
    ```

3. **移除依赖**

- 移除主要（生产）依赖：
    ```bash
    uv remove <package_name>
    # 这会自动更新 pyproject.toml 并移除包
    ```
- 移除开发依赖：
    ```bash
    uv remove <package_name> --group dev
    # 示例：uv remove ruff --group dev
    # 这会从 [project.optional-dependencies.dev] 部分移除包
    ```
- 移除其他类型的可选依赖：
    ```bash
    uv remove <package_name> --group test
    uv remove <package_name> --group docs
    ```

4. **管理环境**

- 使用添加/移除命令后，同步环境：
    ```bash
    uv sync
    ```

**更新日志**
---
本项目使用 [git-changelog](https://github.com/pawamoy/git-changelog)
基于 [Conventional Commits](https://www.conventionalcommits.org/) 自动生成和维护更新日志。

1. **安装**
   ```bash
   pip install git-changelog
   # 或使用 uv 将其添加为开发依赖
   uv add git-changelog --group dev
   ```
2. **验证安装**
   ```bash
   pip show git-changelog
   # 或专门检查版本
   pip show git-changelog | grep Version
   ```
3. **配置**
   确保您在项目根目录有一个正确配置的 `pyproject.toml` 文件。配置应将 Conventional Commits 指定为更新日志样式。以下是示例配置：
   ```toml
   [tool.git-changelog]
   version = "0.1.0"
   style = "conventional-commits"
   output = "CHANGELOG.md"
   ```
4. **生成更新日志**
   ```bash
   git-changelog --output CHANGELOG.md
   # 或者使用 uv 运行
   uv run git-changelog --output CHANGELOG.md
   ```
   此命令会根据您的 git 历史记录创建或更新 `CHANGELOG.md` 文件。
5. **推送更改**
   ```bash
   git push origin main
   ```
   或者，使用您 IDE 的 Git 界面（例如，在许多编辑器中的 `Git → Push`）。
6. **注意**：

- 更新日志是根据您的提交消息遵循 Conventional Commits 规范自动生成的。
- 每当您想要更新更新日志时（通常在发布之前或进行重大更改之后），运行生成命令。
