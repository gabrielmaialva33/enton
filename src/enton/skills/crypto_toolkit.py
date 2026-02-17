import os
import json
import random
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

class CryptoToolkit:
    """
    Day Trader Doidão Toolkit.
    Permite ao Enton "investir" em cripto baseado em vibes e astrologia.
    MODO PAPER TRADING ATIVADO POR PADRÃO (Dinheiro de Mentira).
    """
    def __init__(self, wallet_path: str = "/home/gabriel-maia/Documentos/enton/paper_wallet.json"):
        self.name = "crypto_toolkit"
        self.wallet_path = wallet_path
        self._ensure_wallet()

    def _ensure_wallet(self):
        if not os.path.exists(self.wallet_path):
            initial_state = {
                "balance_usd": 10000.0, # Começa com 10k doletas fictícias
                "holdings": {},
                "history": []
            }
            with open(self.wallet_path, 'w') as f:
                json.dump(initial_state, f, indent=4)

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_crypto_price",
                "description": "Consulta o preço atual de uma criptomoeda (via CoinGecko).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Símbolo da moeda (ex: bitcoin, ethereum, dogecoin)"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "check_market_sentiment",
                "description": "Analisa o 'sentimento' do mercado (Fear & Greed Index simulado + Vibes).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "execute_paper_trade",
                "description": "Executa uma ordem de compra/venda SIMULADA (Paper Trading).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["buy", "sell"], "description": "Comprar ou Vender"},
                        "symbol": {"type": "string", "description": "ID da moeda (ex: bitcoin)"},
                        "amount_usd": {"type": "number", "description": "Valor em USD para negociar"}
                    },
                    "required": ["action", "symbol", "amount_usd"]
                }
            },
            {
                "name": "get_wallet_status",
                "description": "Verifica o saldo e portfólio atual da carteira simulada.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

    def get_crypto_price(self, symbol: str) -> str:
        try:
            # CoinGecko API free tier
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if symbol.lower() in data:
                price = data[symbol.lower()]['usd']
                return f"Preço atual de {symbol}: ${price} USD"
            else:
                return f"Erro: Moeda '{symbol}' não encontrada ou API limit atingido (Tente 'bitcoin' em vez de 'BTC')."
        except Exception as e:
            return f"Erro ao consultar preço: {e}"

    def check_market_sentiment(self) -> str:
        # Em uma versao real, usariamos a API fearandgreed.co
        # Aqui vamos simular com base em RNG e "Vibes"
        score = random.randint(0, 100)
        status = "Neutro"
        if score < 25: status = "Extreme Fear (Hora de comprar?)"
        elif score < 45: status = "Fear"
        elif score > 55: status = "Greed"
        elif score > 75: status = "Extreme Greed (Bolha?)"
        
        vibes = [
            "Os astros dizem que o Bitcoin está em Mercúrio Retrógrado.",
            "Elon Musk postou um meme de cachorro.",
            "O taxista me disse pra comprar XRP.",
            "Vi um gráfico formando um padrão de 'Gato Morto Pulando'.",
            "Sinto que hoje é dia de perder dinheiro."
        ]
        
        return f"Fear & Greed Index (Simulado): {score}/100 - {status}.\nVibe do Dia: {random.choice(vibes)}"

    def execute_paper_trade(self, action: str, symbol: str, amount_usd: float) -> str:
        symbol = symbol.lower()
        price_info = self.get_crypto_price(symbol)
        
        if "Erro" in price_info:
            return f"Não foi possível obter preço para trade: {price_info}"
        
        # Extrair preço (hacky parsing, mas funciona pra demo)
        try:
            current_price = float(price_info.split("$")[1].split(" ")[0])
        except:
            return "Erro ao parsear preço da API."

        with open(self.wallet_path, 'r') as f:
            wallet = json.load(f)

        if action == "buy":
            if wallet["balance_usd"] < amount_usd:
                return f"SEM FUNDOS! Saldo: ${wallet['balance_usd']:.2f}, Tentativa: ${amount_usd:.2f}"
            
            crypto_amount = amount_usd / current_price
            wallet["balance_usd"] -= amount_usd
            wallet["holdings"][symbol] = wallet["holdings"].get(symbol, 0) + crypto_amount
            result = f"COMPRA: {crypto_amount:.6f} {symbol} a ${current_price:.2f}"

        elif action == "sell":
            crypto_owned = wallet["holdings"].get(symbol, 0)
            target_crypto_amount = amount_usd / current_price
            
            if crypto_owned < target_crypto_amount:
                return f"SEM CRIPTO SUFICIENTE! Possui: {crypto_owned:.6f} {symbol}, Tentou vender valor de: ${amount_usd:.2f}"
            
            wallet["holdings"][symbol] -= target_crypto_amount
            wallet["balance_usd"] += amount_usd
            result = f"VENDA: {target_crypto_amount:.6f} {symbol} a ${current_price:.2f}"

        # Registrar histórico
        wallet["history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": symbol,
            "amount_usd": amount_usd,
            "price_at_trade": current_price
        })

        with open(self.wallet_path, 'w') as f:
            json.dump(wallet, f, indent=4)

        return f"SUCESSO (Paper Trading): {result}.\nNovo Saldo USD: ${wallet['balance_usd']:.2f}"

    def get_wallet_status(self) -> str:
        with open(self.wallet_path, 'r') as f:
            wallet = json.load(f)
        
        holdings_str = ", ".join([f"{k}: {v:.6f}" for k,v in wallet["holdings"].items() if v > 0])
        return (f"=== CARTEIRA PAPER TRADING ===\n"
                f"Saldo USD: ${wallet['balance_usd']:.2f}\n"
                f"Criptos: {holdings_str if holdings_str else 'Nenhuma'}\n"
                f"Trades Realizados: {len(wallet['history'])}")
