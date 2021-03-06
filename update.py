import feedparser
from bs4 import BeautifulSoup #导入bs4库
import urllib.request
import sys
import os
import time
import sqlite3
def parse_url(str1,file, folder):
  data = feedparser.parse(str1)
  sql_path ='.database/'+folder+'.sqlite3'
  if os.path.exists(sql_path):
    conn = sqlite3.connect(sql_path)
    cur = conn.cursor()
  else:
    conn = sqlite3.connect(sql_path)
    cur = conn.cursor()
    sql_text_1 = '''CREATE TABLE bookmark 
              ( ID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                COUNT_NUM INTERGER,
                URL TEXT,
                RAW_DATA TEXT,
                Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);'''
    cur.execute(sql_text_1)
  
  for entry in data.entries:
    url = entry.link
    exe2= "SELECT * FROM bookmark WHERE URL == " + '"'+url+'"'
    result_cursor = cur.execute(exe2)
    result = result_cursor.fetchall()
    if len(result) == 0:
      headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
      req=urllib.request.Request(url, headers=headers)
      article_title=''
      try:
        resp=urllib.request.urlopen(req)
      except :
        print( 'We failed to reache a server on ' + url)
      else:
        raw_soup=resp.read().decode('utf-8','ignore')
        soup = BeautifulSoup(raw_soup,'lxml')
        try:
          article_title = str("### [" +str(soup.title.get_text())+"]("+url+")")
          abstract = soup.find('blockquote', class_='abstract mathjax')
          abstract.span.extract()
        except:
          article_title = str("### [" +str(soup.title)+"]("+url+")")
        file.writelines('\n\n'+article_title)
        if abstract is not None:
          file.writelines('\n\n'+abstract.text)
      if len(article_title) > 0:
        exe1 = 'INSERT INTO bookmark VALUES(null,1,"'+url+'","'+article_title.replace('"',' ')+'", (strftime("%Y-%m-%d %H:%M:%f","now", "localtime")))'
        print(exe1)
        cur.execute(exe1)
        conn.commit()
  conn.close()
      
def update_cal(path, localtime):
  readme = open(path+"/bookmark.md","a")
  readme.write(" ["+str(localtime.tm_mon)+"-"+str(localtime.tm_mday)+"](./"+str(localtime.tm_year)+"/"+str(localtime.tm_mon)+"-"+str(localtime.tm_mday)+".md) |")
  if localtime.tm_wday == 6:
    readme.write("\n|")

def main(argv):
  localtime = time.localtime(time.time())
  file_name =argv[1]+'/'+str(localtime.tm_year)+"/"+str(localtime.tm_mon)+'-'+str(localtime.tm_mday)+".md"
  is_exists = os.path.exists(file_name)
  file=open(file_name,"a")
  if not is_exists:
    file.write("\n## "+str(localtime.tm_year)+"-"+str(localtime.tm_mon)+"-"+str(localtime.tm_mday))
  url_list = open(argv[1]+'/list.txt','r').read().splitlines()
  for url in url_list:
    parse_url(url, file, argv[1])
  file.close()
  if not is_exists:
    update_cal(argv[1], localtime)
  
if __name__ == "__main__":
  main(sys.argv)
