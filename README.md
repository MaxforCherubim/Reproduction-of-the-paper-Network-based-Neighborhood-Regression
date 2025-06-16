# 论文《Network-based Neighborhood Regression》复现

[![Template](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/roaldarbol/r-template//main/badge.json)](https://github.com/roaldarbol/r-template)

arXiv:2407.04104

## 目录

- [论文《Network-based Neighborhood Regression》复现](#论文network-based-neighborhood-regression复现)
  - [目录](#目录)
  - [项目概述](#项目概述)
  - [使用方法](#使用方法)
  - [注意事项](#注意事项)
  - [贡献指南​](#贡献指南)
  - [联系方式​](#联系方式)
  - [许可证信息​](#许可证信息)

## 项目概述

- 本项目为论文[《Network-based Neighborhood Regression》](https://arxiv.org/abs/2407.04104)的复现
- 详细说明了项目复现过程中遇到的各种问题，并提出了解决方案
- 本项目基于Python和R语言，使用[pixi](pixi.sh)管理依赖和工作空间,同时配置了Windows和Mac系统的运行环境，基本上做到了开箱即用
- 本项目同时还使用了[R Pixi template](https://github.com/roaldarbol/r-template)的模板，使得R语言的配置为pixi下的R语言环境配置

> [!TIP]
> <img src="https://pixi.sh/latest/assets/pixi.png" width="50">Pixi是用Rust编写的一个环境管理工具，极度轻量化同时速度极快；更大的优势是同时集成了R、Python、Node.js等语言的环境管理，可以方便地管理不同语言的依赖和工作空间。

## 使用方法

1. 参考[pixi安装方法](https://pixi.sh/latest/installation/)，在本地安装好pixi
2. 克隆项目到本地：`git clone https://github.com/bay23/network-based-neighborhood-regression.git`
3. cd到本项目中，运行`pixi install`即可同时配置好Python和R语言的环境
4. 本项目已经配置好了两个代码编辑器，其他编辑器视情况需要使用者自行配置
   - 使用Visual Studio Code编辑器：`pixi run code`
   - 使用JupyterLab编辑器：`pixi run jupyter lab`
5. 启动好编辑器后，即可运行复现代码

## 注意事项

1. 由于本项目存在需要手动编译安装的R包，所以必须要按照对应的Rtools，本项目Rtools版本为4.0
    - 在R终端中运行命令`install.Rtools()`，一般会自动根据本项目的R版本安装对应的4.0版本
    - Rtools40安装完成后，需要运行命令`Sys.which('make')`检测是否安装成功，返回空字符串则安装失败
    - 我在[.Renviron](./.Renviron)文件中添加了正确路径，一般会显示安装成功
    - 还需要运行命令`Sys.which('g++')`检测g++等相关工具是否添加到路径中，如果该命令返回空字符串，则需要手动将g++上级目录添加到环境变量中，类似该路径`C:/Tools/Rtools/rtools40/mingw64/bin`；添加环境变量请自行百度
2. netcoh包0.35版本并不在[CRAN](https://cran.r-project.org/)上，而是在 https://github.com/tianxili/RNC 这个项目中，因此需要手动安装
    - 下载[netcoh_0.35.tar.gz](./netcoh_0.35.tar.gz)文件到项目中，当然本项目自带该文件放在项目根目录中
    - 安装命令为`install.packages('./netcoh_0.35.tar.gz', repos=NULL, type='source', dependencies = TRUE)`
3. 解决了上述两个问题大概率还不能使得代码成功运行，还需要将pixi环境下的R路径添加到环境变量中，即将该路径下[.pixi/envs/default/lib/R/bin/x64](./.pixi/envs/default/lib/R/bin/x64)的绝对路径添加到环境变量中
4. 最后一个bug是修改[.pixi/envs/default/lib/site-packages/rpy2/rinterface_lib/conversion.py](.pixi/envs/default/lib/site-packages/rpy2/rinterface_lib/conversion.py)文件中的`_cchar_to_str`和`_cchar_to_str_with_maxlen`函数
    
    将原始代码的
    ```python
    s = ffi.string(c).decode(encoding)
    ```
    修改为
    ```python
    try:
        s = ffi.string(c).decode(encoding)
    except Exception as e:
        s = ffi.string(c).decode('GBK')
    ```

## 贡献指南​

- 本项目强烈呼吁各位使用者提交反馈，包括但不限于在使用过程中遇到的问题、想到的建议，甚至是小白问题都可以在讨论区创建讨论，我会及时参与讨论的！**我们的所有互动都会成为后来者的学习资料，请积极参加！**
- 如果对本项目感兴趣，甚至想要进一步优化，欢迎提交PR！

## 联系方式​

- 作者：[章迎潭](https://github.com/MaxforCherubim)
- 导师：[马海强教授](https://stat.jxufe.edu.cn/news-show-7166.html)
- 邮件：<EMAIL>bay237580157@outlook.com</EMAIL>

## 许可证信息​

本项目开源许可证为[MIT license]

不排除涉及论文作者权益导致后续修改

[回到顶部](#目录)
