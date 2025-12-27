#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/14 23:18
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :   

from app.tools.layout import config_page, set_pages


def main() -> None:
    """ streamlit run main.py """
    config_page()
    set_pages()


if __name__ == "__main__":
    main()
