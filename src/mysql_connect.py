import mysql.connector as mysql
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

connection = mysql.connect(
    host=os.getenv("host"),
    user='root',
    password=os.getenv("password"),
    database=os.getenv('database'),
    auth_plugin='mysql_native_password'
)

cursor = connection.cursor()

sql = "SELECT * FROM windpower.windpower;"
cursor.execute(sql)

data = cursor.fetchall()

columns = ['Date_Time', 'LV_ActivePower_kW', 'WindSpeed_m_per_s',
           'Theoretical_Power_Curve_kilowatt_hour', 'Wind_Direction_degree']

df = pd.DataFrame(data=data, columns=columns)