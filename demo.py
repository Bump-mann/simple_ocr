import requests

headers = {
    'authority': '810fk.cn',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'zh-CN,zh;q=0.9,en-GB;q=0.8,en-US;q=0.7,en;q=0.6',
    'cache-control': 'no-cache',
    'cookie': 's9d38f919=mgqhhma3miji883oo6p0d2p5v0; clicaptcha_text=%E9%AA%97%2C%E6%9B%B4%2C%E6%97%8B',
    'pragma': 'no-cache',
    'referer': 'https://810fk.cn/orderquery',
    'sec-ch-ua': '".Not/A)Brand";v="99", "Google Chrome";v="103", "Chromium";v="103"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
}

params = (
    ('orderid', '738389368'),
    ('chkcode', 'EVoP5Gz46Nu9QD393rFDqPIAoQbsnvt4BwPhLtp0tQ8='),
    ('querytype', '3'),
)

response = requests.get('https://810fk.cn/orderquery', headers=headers, params=params)
print(response.text)
#NB. Original query string below. It seems impossible to parse and
#reproduce query strings 100% accurately so the one below is given
#in case the reproduced version is not "correct".
# response = requests.get('https://810fk.cn/orderquery?orderid=738389368&chkcode=EVoP5Gz46Nu9QD393rFDqPIAoQbsnvt4BwPhLtp0tQ8%3D&querytype=3', headers=headers)
