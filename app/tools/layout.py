#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:18
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   layout.py
# @Desc     :

from streamlit import set_page_config, Page, navigation


def config_page() -> None:
    """ Set the window
    :return: None
    """
    set_page_config(
        page_title="RNN Translater 1.0",
        page_icon=":material/globe:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def set_pages() -> None:
    """ Set the subpages on the sidebar
    :return: None
    """
    pages: dict = {
        "page": [
            "app/subpages/home.py",
            "app/subpages/prediction.py",
        ],
        "title": [
            "Home",
            "Prediction",
        ],
        "icon": [
            ":material/home:",
            ":material/function:",
        ],
    }

    structure: dict = {
        "Introduction": [
            Page(page=pages["page"][0], title=pages["title"][0], icon=pages["icon"][0]),
        ],
        "Core Functions": [
            Page(page=pages["page"][1], title=pages["title"][1], icon=pages["icon"][1]),
        ],
    }
    pg = navigation(structure, position="sidebar", expanded=True)
    pg.run()
