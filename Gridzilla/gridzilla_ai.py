"""
Gridzilla AI Signal Engine - Multiprocessing Edition
Isolates the LLM in a separate process to prevent TUI freezes.
"""
import os
import json
import time
import multiprocessing
from decimal import Decimal
from typing import Optional, Dict

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "unsloth-r1-7b", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")

SYSTEM_PROMPT = "You are a Senior Quantitative Trading Analyst. Your task is to analyze market data, global sentiment, and recent performance to issue a final trade decision. Output JSON only. Format: { \"decision\": \"BUY\"/\"SELL\"/\"HOLD\", \"confidence\": 0-100, \"grade\": \"STANDARD\"/\"HIGH\"/\"ULTRA\", \"rationale\": \"brief logic\", \"risk_assessment\": \"low\"/\"medium\"/\"high\" }. RULES: 1. If Global Sentiment < 40%, be extremely conservative (Risk-Off). 2. If Recent Performance shows 3+ consecutive losses, prioritize HOLD unless confidence is 95%+. 3. Ignore RSI if Volume is 3x the 24h average (Volume overrides overbought). 4. Output ONLY valid JSON."

def ai_worker(input_queue, output_queue, model_path, n_ctx, n_threads):
    """The isolated process that runs the LLM."""
    try:
        from llama_cpp import Llama
        
        # Load the model inside the child process
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        
        # Tell the parent we are ready
        output_queue.put({"status": "READY"})
        
        while True:
            # Wait for a request
            request = input_queue.get()
            if request == "SHUTDOWN":
                llm.close()
                break
            
            symbol = request["symbol"]
            prompt = request["prompt"]
            
            start_time = time.time()
            
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            
            elapsed = time.time() - start_time
            content = response["choices"][0]["message"]["content"]
            
            output_queue.put({
                "symbol": symbol,
                "content": content,
                "elapsed": elapsed
            })
            
    except Exception as e:
        output_queue.put({"status": "ERROR", "error": str(e)})

class GridzillaAI:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or MODEL_PATH
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = None
        self.is_ready = False
        
    def start(self):
        """Spawn the AI process."""
        if self.process and self.process.is_alive():
            return True
            
        self.process = multiprocessing.Process(
            target=ai_worker,
            args=(self.input_queue, self.output_queue, self.model_path, 4096, 1),
            daemon=True
        )
        self.process.start()
        return True

    def check_status(self):
        """Non-blocking check for ready status or results."""
        try:
            while not self.output_queue.empty():
                msg = self.output_queue.get_nowait()
                if msg.get("status") == "READY":
                    self.is_ready = True
                    return "READY", None
                if msg.get("status") == "ERROR":
                    return "ERROR", msg.get("error")
                return "RESULT", msg
        except:
            pass
        return "WAITING", None

    def request_analysis(self, symbol, market_data, global_data=None, history=None):
        """Send a request to the AI process."""
        if not self.is_ready:
            return False
            
        prompt = self._build_prompt(symbol, market_data, global_data, history)
        self.input_queue.put({"symbol": symbol, "prompt": prompt})
        return True

    def _build_prompt(self, symbol: str, market_data: dict, global_data: dict = None, history: list = None) -> str:
        """Format market data into prompt with Global Sentiment and History."""
        
        # Format Performance History (Memory)
        hist_str = "No recent trade data."
        if history:
            items = []
            for h in history[-5:]: # Last 5 trades
                res = "WIN" if h.get('r_multiple', 0) > 0 else "LOSS"
                items.append(f"{h.get('pair', '???')}: {res} ({float(h.get('r_multiple', 0)):+.1f}R)")
            hist_str = " | ".join(items)

        # Format Global Sentiment (Market Breath)
        glob_str = "Sentiment Data Unavailable."
        if global_data:
            bull_pct = global_data.get('bullish_pct', 50)
            btc_trend = global_data.get('btc_trend', 'Neutral')
            glob_str = f"Market Breath: {bull_pct}% Bullish | BTC Trend: {btc_trend}"

        data = {
            "symbol": symbol,
            "price": market_data.get("price_cur", 0),
            "rsi": market_data.get("rsi", 50),
            "macd_line": market_data.get("macd_line", 0),
            "macd_signal": market_data.get("macd_signal", 0),
            "macd_hist": market_data.get("macd_hist", 0),
            "stoch_k": market_data.get("stoch_k", 50),
            "stoch_d": market_data.get("stoch_d", 50),
            "ema_fast": market_data.get("ema_fast", 0),
            "ema_slow": market_data.get("ema_slow", 0),
            "supertrend_dir": market_data.get("supertrend_dir", False),
            "supertrend_val": market_data.get("supertrend_val", 0),
            "vo": market_data.get("vo", 0),
            "atr": market_data.get("atr", 0),
            "bbw": market_data.get("bbw", 0),
            "bbw_avg": market_data.get("bbw_avg", 0),
            "support": market_data.get("floor", 0),
            "resistance": market_data.get("ceil", 0),
            "fib_50": market_data.get("fib_50", 0),
            "fib_618": market_data.get("fib_618", 0),
            "volume_rank": market_data.get("volume_rank", 999),
            "4h_ema_20": market_data.get("ema_macro_f", 0),
            "4h_ema_50": market_data.get("ema_macro_s", 0),
            "bullish_1h": market_data.get("bullish", False),
            "signals_fired": market_data.get("signals_fired", []),
            "signal_count": market_data.get("signal_count", 0),
        }
        
        prompt = f"""### TRADING ANALYSIS REQUEST: {symbol}
### 1. GLOBAL MARKET SENTIMENT (CONTEXT)
{glob_str}

### 2. RECENT BOT PERFORMANCE (MEMORY)
{hist_str}

### 3. INDIVIDUAL PAIR DATA ({symbol})
- Price: ${data['price']} | RSI(14): {data['rsi']}
- MACD: line={data['macd_line']}, signal={data['macd_signal']}, hist={data['macd_hist']}
- Stochastic: %K={data['stoch_k']}, %D={data['stoch_d']}
- EMA(13): {data['ema_fast']} | EMA(33): {data['ema_slow']}
- SuperTrend: {'Bullish' if data['supertrend_dir'] else 'Bearish'} ({data['supertrend_val']})
- Volume Oscillator: {data['vo']}% | ATR: {data['atr']}
- BBW: {data['bbw']} (avg: {data['bbw_avg']})
- Support: {data['support']} | Resistance: {data['resistance']}
- Fib 0.5: {data['fib_50']} | Fib 0.618: {data['fib_618']}
- 4h EMA(20): {data['4h_ema_20']} | 4h EMA(50): {data['4h_ema_50']}

### 4. TECHNICAL SIGNALS
- Fired: {data['signals_fired']} ({data['signal_count']}/11)

Respond with JSON only."""
        return prompt

    def parse_result(self, msg):
        """Parse the AI content into a decision dict."""
        content = msg["content"]
        elapsed = msg["elapsed"]
        
        # Robust JSON extraction
        cleaned_content = content
        if "<think>" in content and "</think>" in content:
            cleaned_content = content.split("</think>")[-1].strip()
        elif "</think>" in content:
            cleaned_content = content.split("</think>")[-1].strip()
        
        json_start = cleaned_content.find('{')
        json_end = cleaned_content.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            try:
                json_str = cleaned_content[json_start:json_end]
                result = json.loads(json_str)
                result["_elapsed"] = elapsed
                return result
            except:
                pass
        
        # Fallback
        content_lower = content.lower()
        decision = "HOLD"
        confidence = 50
        if "buy" in content_lower: decision = "BUY"; confidence = 75
        
        return {
            "decision": decision, 
            "confidence": confidence, 
            "grade": "STANDARD", 
            "_elapsed": elapsed
        }

    def stop(self):
        """Shutdown the AI process."""
        self.input_queue.put("SHUTDOWN")
        if self.process:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()

# Global instance
ai_instance = GridzillaAI()

def initialize_model():
    return ai_instance.start()

def request_analysis(symbol, data):
    return ai_instance.request_analysis(symbol, data)

def check_results():
    status, data = ai_instance.check_status()
    if status == "RESULT":
        return ai_instance.parse_result(data)
    return status, data

def shutdown_ai():
    ai_instance.stop()
