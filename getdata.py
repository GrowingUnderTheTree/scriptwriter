import requests
from bs4 import BeautifulSoup
import pandas as pd
links = ["https://imsdb.com/scripts/Batman.html",
"https://imsdb.com/scripts/Batman-2.html",
"https://imsdb.com/scripts/Dark-Knight-Rises,-The.html",
"https://imsdb.com/scripts/Joker.html",
"https://imsdb.com/scripts/Supergirl.html",
"https://imsdb.com/scripts/Spider-Man.html",
"https://imsdb.com/scripts/Apocalypse-Now.html",
"https://imsdb.com/scripts/Blitz.html",
"https://imsdb.com/scripts/Battle-of-Algiers,-The.html",
"https://imsdb.com/scripts/Barry-Lyndon.html",
"https://imsdb.com/scripts/Jojo-Rabbit.html",
"https://imsdb.com/scripts/Pearl-Harbor.html",
"https://imsdb.com/scripts/John-Wick.html",
"https://imsdb.com/scripts/John-Wick-Chapter-4.html",
"https://imsdb.com/scripts/Ladykillers,-The.html",
"https://imsdb.com/scripts/L.A.-Confidential.html",
"https://imsdb.com/scripts/Lone-Star.html",
"https://imsdb.com/scripts/Looper.html",
"https://imsdb.com/scripts/Lord-of-War.html",
"https://imsdb.com/scripts/Losers,-The.html",
"https://imsdb.com/scripts/Machine-Gun-Preacher.html",
"https://imsdb.com/scripts/Alien.html",
"https://imsdb.com/scripts/Alien-3.html",
"https://imsdb.com/scripts/Alien-Nation.html",
"https://imsdb.com/scripts/Alien-vs.-Predator.html",
"https://imsdb.com/scripts/Alien-Resurrection.html",
"https://imsdb.com/scripts/Aliens.html",
"https://imsdb.com/scripts/American-Psycho.html",
"https://imsdb.com/scripts/Avatar.html",
"https://imsdb.com/scripts/Avengers,-The.html",
"https://imsdb.com/scripts/Avengers,-The-(2012).html",
"https://imsdb.com/scripts/Avengers-Endgame.html",
"https://imsdb.com/scripts/Beauty-and-the-Beast.html",
"https://imsdb.com/scripts/Blade.html",
"https://imsdb.com/scripts/Blade-II.html",
"https://imsdb.com/scripts/Blade-Runner.html",
"https://imsdb.com/scripts/Blade-Trinity.html"]
data = []
for i in range(len(links)):
    response = requests.get(links[i])
    soup = BeautifulSoup(response.text, "html.parser")
    element = soup.find(class_='scrtext').get_text()
    data.append(element)
df = pd.DataFrame(data)
df.to_csv('scripts.csv', index=False)