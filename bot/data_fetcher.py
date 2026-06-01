import os
import requests
import pandas as pd

POLYMARKET_BASE = 'https://clob.polymarket.com'
POLYGON_BASE = 'https://api.polygon.io'


class PolymarketFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})

    def get_markets(self, limit=50):
        resp = self.session.get(f'{POLYMARKET_BASE}/markets', params={'limit': limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get('data', data) if isinstance(data, dict) else data

    def get_orderbook(self, token_id):
        resp = self.session.get(f'{POLYMARKET_BASE}/book', params={'token_id': token_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def to_dataframe(self, markets):
        rows = []
        for m in markets:
            tokens = m.get('tokens', [])
            yes_token = next((t for t in tokens if t.get('outcome') == 'Yes'), None)
            rows.append({
                'market_id': m.get('condition_id'),
                'question': m.get('question'),
                'price': float(yes_token.get('price', 0)) if yes_token else None,
                'volume': float(m.get('volume', 0)),
                'liquidity': float(m.get('liquidity', 0)),
                'end_date': m.get('end_date_iso'),
            })
        return pd.DataFrame(rows).dropna(subset=['price'])


class PolygonFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('POLYGON_API_KEY', '')
        if not self.api_key:
            raise ValueError('Set POLYGON_API_KEY in your .env file.')
        self.session = requests.Session()

    def _get(self, endpoint, params={}):
        params['apiKey'] = self.api_key
        resp = self.session.get(f'{POLYGON_BASE}{endpoint}', params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_daily_prices(self, ticker, from_date, to_date):
        data = self._get(f'/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}')
        df = pd.DataFrame(data.get('results', []))
        if not df.empty:
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        return df


if __name__ == '__main__':
    fetcher = PolymarketFetcher()
    try:
        markets = fetcher.get_markets(limit=10)
        df = fetcher.to_dataframe(markets)
        print(f'Fetched {len(df)} markets')
        print(df[['question', 'price', 'volume']].head())
    except Exception as e:
        print(f'Error: {e}')
