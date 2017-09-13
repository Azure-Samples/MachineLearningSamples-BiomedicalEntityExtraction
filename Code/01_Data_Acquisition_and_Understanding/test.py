pip install unidecode

import requests
import lxml
import unidecode 
import pubmed_parser as pp

x = 'C:\Users\hacker\Downloads\medline17n0001.xml.gz'

output = pp.parse_medline_xml(x)