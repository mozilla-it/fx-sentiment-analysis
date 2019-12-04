
import requests

def get_csv_from_url(url,filename):
    with requests.Session() as s:
        d = s.get(url)
        c = d.content
        with open(filename,'wb') as w:
            w.write(c)
