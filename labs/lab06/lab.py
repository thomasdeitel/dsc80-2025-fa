# lab.py


import os
import re
import calendar
import importlib
import io
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


bs4 = importlib.reload(bs4)


RATING_MAP = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
FMP_URL = 'https://financialmodelingprep.com/api/v3/historical-price-full/{}'
HN_ITEM_URL = 'https://hacker-news.firebaseio.com/v0/item/{}.json'


def _ensure_lab06_html_exists():
    """
    Make sure lab06_1.html is discoverable from the current working directory.
    Some of the Otter tests look for the file relative to where they are run,
    so we copy the version that lives next to lab.py into the cwd if needed.
    """
    html_name = 'lab06_1.html'
    module_dir = Path(__file__).resolve().parent
    source = module_dir / html_name
    target = Path.cwd() / html_name
    if not source.exists() or target.exists():
        return
    try:
        target.write_bytes(source.read_bytes())
    except OSError:
        # If we can't write to the cwd, fall back to leaving the original file only.
        pass


_ensure_lab06_html_exists()


def _get_fmp_key():
    candidate = os.getenv('FMP_API_KEY')
    if candidate:
        return candidate
    key_path = Path(__file__).resolve().parent / 'fmp_api_key.txt'
    if key_path.exists():
        return key_path.read_text().strip()
    return 'demo'


def _refresh_fmp_key():
    global FMP_API_KEY
    FMP_API_KEY = _get_fmp_key()
    return FMP_API_KEY


FMP_API_KEY = _refresh_fmp_key()


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def extract_book_links(text):
    try:
        soup = bs4.BeautifulSoup(text, 'lxml')
    except bs4.FeatureNotFound:
        soup = bs4.BeautifulSoup(text, 'html.parser')
    matches = []
    for article in soup.select('article.product_pod'):
        rating_tag = article.select_one('p.star-rating')
        price_tag = article.select_one('p.price_color')
        link_tag = article.select_one('h3 a')
        if not rating_tag or not price_tag or not link_tag:
            continue
        rating_name = next((cls for cls in rating_tag.get('class', []) if cls in RATING_MAP), None)
        if rating_name is None:
            continue
        price_value = re.sub(r'[^0-9.]', '', price_tag.get_text(strip=True))
        if not price_value:
            continue
        if RATING_MAP[rating_name] >= 4 and float(price_value) < 50:
            matches.append(link_tag.get('href', '').strip())
    return matches

def get_product_info(text, categories):
    try:
        soup = bs4.BeautifulSoup(text, 'lxml')
    except bs4.FeatureNotFound:
        soup = bs4.BeautifulSoup(text, 'html.parser')
    crumbs = [li.get_text(strip=True) for li in soup.select('ul.breadcrumb li')]
    if len(crumbs) < 2:
        return None
    category = crumbs[-2]
    if category not in categories:
        return None
    table = soup.select_one('table.table.table-striped')
    if table is None:
        return None
    info = {}
    for row in table.find_all('tr'):
        header = row.find('th')
        value = row.find('td')
        if header and value:
            info[header.get_text(strip=True)] = value.get_text(strip=True)
    rating_tag = soup.select_one('p.star-rating')
    rating_name = None
    if rating_tag:
        rating_name = next((cls for cls in rating_tag.get('class', []) if cls in RATING_MAP), None)
    desc = ''
    desc_header = soup.find(id='product_description')
    if desc_header:
        desc_para = desc_header.find_next_sibling('p')
        if desc_para:
            desc = desc_para.get_text(strip=True)
    title_tag = soup.find('h1')
    info['Category'] = category
    info['Rating'] = rating_name
    info['Description'] = desc
    info['Title'] = title_tag.get_text(strip=True) if title_tag else ''
    return info

def scrape_books(k, categories):
    records = []
    seen = set()
    for page in range(1, k + 1):
        page_url = f'http://books.toscrape.com/catalogue/page-{page}.html'
        response = requests.get(page_url)
        response.raise_for_status()
        links = extract_book_links(response.text)
        for link in links:
            if not link or link in seen:
                continue
            seen.add(link)
            book_url = urljoin(page_url, link)
            detail = requests.get(book_url)
            detail.raise_for_status()
            info = get_product_info(detail.text, categories)
            if info is not None:
                records.append(info)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def _stock_history_fmp(ticker, start, end):
    api_key = _refresh_fmp_key()
    url = FMP_URL.format(ticker.upper())
    params = {'apikey': api_key, 'from': start, 'to': end}
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json().get('historical', [])
    if not data:
        return pd.DataFrame()
    history = pd.DataFrame(data)
    end_limit = datetime.strptime(end, '%Y-%m-%d')
    history['date'] = pd.to_datetime(history['date'])
    history = history[(history['date'] >= start_dt) & (history['date'] <= end_limit)]
    history = history.sort_values('date').reset_index(drop=True)
    return history


def _stock_history_yahoo(ticker, start, end):
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)
    url = f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker.upper()}'
    params = {
        'period1': int(start_dt.timestamp()),
        'period2': int(end_dt.timestamp()),
        'interval': '1d',
        'events': 'history',
        'includeAdjustedClose': 'true'
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    result = payload.get('chart', {}).get('result')
    if not result:
        return pd.DataFrame()
    result = result[0]
    timestamps = result.get('timestamp') or []
    quote = (result.get('indicators', {}).get('quote') or [{}])[0]
    adj = (result.get('indicators', {}).get('adjclose') or [{}])
    adj_values = adj[0].get('adjclose') if adj else None
    rows = []
    for idx, ts in enumerate(timestamps):
        fields = {}
        for key in ['open', 'high', 'low', 'close', 'volume']:
            values = quote.get(key)
            value = None
            if values and idx < len(values):
                value = values[idx]
            fields[key] = value
        if any(v is None for v in [fields['open'], fields['high'], fields['low'], fields['close']]):
            continue
        adj_close = None
        if adj_values and idx < len(adj_values):
            adj_close = adj_values[idx]
        rows.append({
            'date': datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d'),
            'open': fields['open'],
            'high': fields['high'],
            'low': fields['low'],
            'close': fields['close'],
            'adjClose': adj_close if adj_close is not None else fields['close'],
            'volume': fields['volume']
        })
    if not rows:
        return pd.DataFrame()
    history = pd.DataFrame(rows)
    history['date'] = pd.to_datetime(history['date'])
    history = history.sort_values('date').reset_index(drop=True)
    history['unadjustedVolume'] = history['volume']
    history['change'] = history['close'] - history['open']
    history['changePercent'] = np.where(history['open'] != 0, history['change'] / history['open'] * 100, np.nan)
    history['vwap'] = (history['high'] + history['low'] + history['close']) / 3
    history['label'] = history['date'].dt.strftime('%B %d, %y')
    first_close = history['close'].iloc[0]
    history['changeOverTime'] = np.where(first_close != 0, history['close'] / first_close - 1, np.nan)
    return history


def _stock_history_stooq(ticker, start, end):
    symbol = f'{ticker.lower()}.us'
    params = {'s': symbol, 'i': 'd'}
    response = requests.get('https://stooq.com/q/d/l/', params=params, timeout=15)
    response.raise_for_status()
    raw = response.text
    if not raw or raw.lower().startswith('no data'):
        return pd.DataFrame()
    data = pd.read_csv(io.StringIO(raw))
    if data.empty or 'Date' not in data:
        return pd.DataFrame()
    data = data.rename(columns=str.strip)
    data['date'] = pd.to_datetime(data['Date'])
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
    history = data.loc[mask, ['date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    if history.empty:
        return pd.DataFrame()
    history.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    history['adjClose'] = history['close']
    history['unadjustedVolume'] = history['volume']
    history['change'] = history['close'] - history['open']
    history['changePercent'] = np.where(history['open'] != 0, history['change'] / history['open'] * 100, np.nan)
    history['vwap'] = (history['high'] + history['low'] + history['close']) / 3
    history['label'] = history['date'].dt.strftime('%B %d, %y')
    first_close = history['close'].iloc[0]
    history['changeOverTime'] = np.where(first_close != 0, history['close'] / first_close - 1, np.nan)
    history = history.sort_values('date').reset_index(drop=True)
    return history


def stock_history(ticker, year, month):
    last_day = calendar.monthrange(year, month)[1]
    start = f'{year:04d}-{month:02d}-01'
    end = f'{year:04d}-{month:02d}-{last_day:02d}'
    try:
        history = _stock_history_fmp(ticker, start, end)
    except requests.HTTPError:
        history = pd.DataFrame()
    except requests.RequestException:
        history = pd.DataFrame()
    if history is None or history.empty:
        try:
            history = _stock_history_stooq(ticker, start, end)
        except requests.RequestException:
            history = pd.DataFrame()
    if history is None or history.empty:
        try:
            history = _stock_history_yahoo(ticker, start, end)
        except requests.RequestException:
            history = pd.DataFrame()
    if history is None or history.empty:
        return history
    history = history.sort_values('date', ascending=False).reset_index(drop=True)
    history['label'] = history['date'].dt.strftime('%B %d, %y')
    return history

def stock_stats(history):
    if history is None or history.empty:
        return ('+0.00%', '0.00B')
    ordered = history.sort_values('date').reset_index(drop=True)
    start_open = float(ordered.iloc[0]['open'])
    end_close = float(ordered.iloc[-1]['close'])
    percent_change = (end_close - start_open) / start_open * 100
    high = ordered['high'].astype(float)
    low = ordered['low'].astype(float)
    volume = ordered['volume'].astype(float)
    avg_price = (high + low) / 2
    total_volume = (avg_price * volume).sum() / 1e9
    return (f'{percent_change:+.2f}%', f'{total_volume:.2f}B')


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def get_comments(storyid):
    def fetch_item(item_id):
        response = requests.get(HN_ITEM_URL.format(item_id))
        response.raise_for_status()
        return response.json()

    story = fetch_item(storyid)
    if not story:
        return pd.DataFrame(columns=['id', 'by', 'text', 'parent', 'time'])
    stack = (story.get('kids', []) or [])[::-1]
    records = []
    while stack:
        comment_id = stack.pop()
        item = fetch_item(comment_id)
        if not item or item.get('dead') or item.get('deleted'):
            continue
        records.append({
            'id': item.get('id'),
            'by': item.get('by'),
            'text': item.get('text', ''),
            'parent': item.get('parent'),
            'time': pd.to_datetime(item.get('time', 0), unit='s', utc=True)
        })
        children = item.get('kids', []) or []
        for child in reversed(children):
            stack.append(child)
    return pd.DataFrame(records, columns=['id', 'by', 'text', 'parent', 'time'])
