import os
import io
import re
import subprocess
from datetime import datetime
from lxml import html
from dateutil import parser

MEDLINE = 'ftp://ftp.nlm.nih.gov/nlmdata/.medleasebaseline/gz/'
PUBMED_OA = 'ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/'

def get_update_date2(option='medline'):
    """
    Download index page and grab lastest update from Medline
    or Pubmed Open-Access subset

    Parameters
    ----------
    option: str, 'medline' for MEDLINE or 'oa' for Pubmed Open-Access

    Example
    -------
    >> date = get_update_date('medline')
    >> print(date.strftime("%Y_%m_%d"))
    """
    if option == 'medline':
        link = MEDLINE
    elif option == 'oa':
        link = PUBMED_OA
    else:
        link = MEDLINE
        print('Specify either "medline" or "oa" for repository')
        
    print('Inside get_update_date2')
    if os.path.exists('index.html'):
        subprocess.call(['rm', 'index.html'])

    print("download " + link + "index.html")
    subprocess.call(['wget', link + "index.html", "--remote-encoding", "utf-8"])

    print("read the index.html")
    with io.open('index.html','r',encoding='utf8') as f:    
        page = f.read()

    date_all = list()
    tree = html.fromstring(page)
    for e in tree.xpath('body/pre/a'):
        if 'File' in e.tail:
            s = e.tail
            s_remove = re.sub(r'\([^)]*\)', '', s)
            s_remove = re.sub('File', '', s_remove).strip()
            d = parser.parse(s_remove)
            date_all.append(d)
    date = max(date_all) # get lastest update
    
    if os.path.exists('index.html'):
        subprocess.call(['rm', 'index.html'])
    return date

