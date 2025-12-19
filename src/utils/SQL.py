#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 19:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   SQL.py
# @Desc     :


from pathlib import Path
from sqlite3 import connect

from src.configs.cfg_base import CONFIG

WIDTH: int = 64


class SQLiteIII:
    """ SQLiteIII Class for Database """

    def __init__(self, table: str, cols: dict, db_path: Path | str | None = None):
        """ Initialise SQLiteIII Database
        :param table: table name
        :param cols: column names
        :param db_path: database file path
        """
        self._connection = None
        self._cursor = None
        self._db = str(db_path) if db_path else CONFIG.FILEPATHS.SQLITE
        self._table = table
        self._cols = cols

    @staticmethod
    def parse_sql_type(python_type):
        if python_type == int:
            return "INTEGER"
        elif python_type == float:
            return "REAL"
        elif python_type == str:
            return "TEXT"
        else:
            raise ValueError(f"Unsupported Python type: {python_type}")

    def __enter__(self):
        if self._connection is None:
            self._connection = connect(self._db)
            self._cursor = self._connection.cursor()

            # Create a table in database
            parsed_cols = ",\n".join([f"{col} {self.parse_sql_type(tp)} NOT NULL" for col, tp in self._cols.items()])
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {parsed_cols},
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self._cursor.execute(sql)
            self._connection.commit()

            print("*" * WIDTH)
            print("SQLite III")
            print("-" * WIDTH)
            print(f"Congratulations! {self._table} table connected and initialised")
            print("*" * WIDTH)
            print()

        return self

    def connect(self):
        """ Connect to the database without using 'with' """
        return self.__enter__()

    def insert(self, data: dict):
        """ Insert data into the database
        :param data: data to insert
        """
        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All columns must have the same number of elements")

        cols = ", ".join(data.keys())
        params = list(zip(*data.values()))
        value_amount: str = ", ".join(["?"] * len(data))

        self._connection.executemany(f"insert into {self._table} ({cols}) values ({value_amount})", params)
        self._connection.commit()

        print(f"{len(params)} rows inserted")

    def count(self) -> int:
        self._cursor.execute(f"select count(*) from {self._table}")

        return self._cursor.fetchone()[0]

    def fetch_all(self, col_names: list[str]) -> list[tuple]:
        names: str = ", ".join(col_names)
        self._cursor.execute(f"select {names} from {self._table} order by id")

        return self._cursor.fetchall()

    def remove_by_id(self, id: int):
        self._cursor.execute(f"delete from {self._table} where id = ?", (id,))
        self._connection.commit()

        deleted = self._cursor.rowcount
        if deleted > 0:
            print(f"Deleted {deleted} row(s)")
        else:
            print(f"No row found with id {id}")

    def search(self, col: str, keyword: str) -> list[str]:
        self._cursor.execute(
            f"select {col} from {self._table} where {col} like ?", (f"%{keyword}%" if keyword else "",)
        )

        return [row[0] for row in self._cursor.fetchall()]

    def clear(self):
        self._cursor.execute(f"delete from {self._table}")
        self._connection.commit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

        return False

    def close(self):
        """ Close database connection manually """
        if self._connection:
            self._connection.close()
            self._connection = None
            self._cursor = None

    @property
    def table_name(self) -> str:
        """ Get the table name """
        return self._table

    @property
    def column_name(self) -> list[str]:
        """ Get the column name """
        return list(self._cols.keys())

    @property
    def database_path(self) -> str:
        """ Get the database file path """
        return self._db


if __name__ == "__main__":
    pass
