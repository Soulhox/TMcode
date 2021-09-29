# In[1]:
import ftplib
from ftplib import FTP
# In[2]:
FTP_HOST = "files.000webhost.com"
FTP_USER = "transmileniosystem"
FTP_PASS = "TransMilenio"
# In[3]:
try:
    ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
    print('CONECTED')
    print(ftp.getwelcome())
except Exception as e:
    print('not connected')
# In[3]:
ftp.retrlines('LIST') 
# In[4]:

#test file, please type the name of the CSV file here
filename = "metricas1.csv"
with open(filename, "rb") as file:
    ftp.storbinary("STOR TMpruebas1.csv", file)
    print("File sent")


