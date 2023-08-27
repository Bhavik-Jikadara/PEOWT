import mysql.connector as mysql
import pandas as pd

connection = mysql.connect(
    host='localhost',
    user='root',
    password='codeB@12j',
    database='windpower',
    auth_plugin='mysql_native_password'
)

cursor = connection.cursor()

sql = "SELECT * FROM windpower.windpower;"
cursor.execute(sql)

data = cursor.fetchall()

columns = ['Date_Time', 'LV_ActivePower_kW', 'WindSpeed_m_per_s',
           'Theoretical_Power_Curve_kilowatt_hour', 'Wind_Direction_degree']

df = pd.DataFrame(data=data, columns=columns)