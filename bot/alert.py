import os
import requests

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID', '')


def send(message, chat_id=None):
    token = BOT_TOKEN
    cid   = chat_id or CHAT_ID

    if not token or not cid:
        print(f'[ALERT] {message}')
        return False

    try:
        resp = requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            json={'chat_id': cid, 'text': message, 'parse_mode': 'Markdown'},
            timeout=5
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f'Telegram failed: {e}')
        return False


def format_ev(opp, bankroll=1000.0):
    bet = round(opp.get('kelly_f', 0) * bankroll, 2)
    return (
        f"*EV Signal*\n"
        f"Market : `{opp.get('market_id', '?')}`\n"
        f"Price  : `{opp.get('price', 0):.1%}` -> Model: `{opp.get('model_p', 0):.1%}`\n"
        f"Net EV : `{opp.get('ev', 0):.3f}` | Kelly: `{opp.get('kelly_f', 0):.2f}`\n"
        f"Bet    : `${bet}` on ${bankroll:.0f} bankroll"
    )


def format_kl(pair):
    return (
        f"*KL Arb Signal*\n"
        f"Pair   : `{pair['market_a']}` vs `{pair['market_b']}`\n"
        f"Prices : `{pair['price_a']:.1%}` vs `{pair['price_b']:.1%}`\n"
        f"KL     : `{pair['kl_score']:.4f}`\n"
        f"Action : hedge both sides"
    )


def format_thin_pool(pool):
    return (
        f"*Thin Pool Detected*\n"
        f"Market : `{pool.get('market_id', '?')}`\n"
        f"b      : `{pool.get('b', 0):.1f}`\n"
        f"Impact : `{pool.get('impact_10', 0):.2%}` per 10 shares"
    )


def format_bregman(arb):
    return (
        f"*Multi-Outcome Arb*\n"
        f"Market : `{arb.get('market_id', '?')}`\n"
        f"Sum    : `{arb.get('price_sum', 0):.4f}`\n"
        f"Type   : `{arb.get('arb_type', '?')}`\n"
        f"Gap    : `{arb.get('deviation', 0):.4f}`"
    )
