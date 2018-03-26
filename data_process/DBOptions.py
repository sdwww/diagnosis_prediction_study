import os
import cx_Oracle as db
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'


# 连接数据库
def connect_db():
    con = db.connect('MH3', '123456', '127.0.0.1:1521/ORCL')
    return con


# 执行select语句
def get_sql(sql, cursor):
    cursor.execute(sql)
    result = cursor.fetchall()
    content = []
    for row in result:
        if len(row) == 1:
            content.append(row[0])
        else:
            content.append(list(row))
    return content


# 执行更新操作
def exe_sql(sql, cursor, con):
    cursor.execute(sql)
    con.commit()
    return 1
