import requests

url = 'http://192.168.29.121:5000/predict'
files = {'file': open('data/demo_imgs/1.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.status_code)
print(response.json())