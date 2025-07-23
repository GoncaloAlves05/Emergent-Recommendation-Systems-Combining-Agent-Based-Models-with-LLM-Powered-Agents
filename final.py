from mesa import Agent, Model
from mesa.time import StagedActivation
from llama_cpp import Llama
import random
from datetime import datetime
import csv
import os
import numpy as np

STEP_FILE = "step_counter.txt"
CSV_FILE = "market_results.csv"

def get_last_step():
    if os.path.exists(STEP_FILE):
        with open(STEP_FILE, "r") as f:
            return int(f.read().strip()) + 1
    return 0

def save_current_step(step):
    with open(STEP_FILE, "w") as f:
        f.write(str(step))

# --- GERADOR DE PREÇOS COM GBM ---
def generate_gbm_price_history(start_price=150, steps=10, mu=0.002, sigma=0.03):
    """
    Gera uma lista de preços usando Geometric Brownian Motion (GBM).
    - start_price: preço inicial
    - steps: número de pontos no histórico
    - mu: taxa de crescimento média
    - sigma: volatilidade
    """
    prices = [float(start_price)]
    for _ in range(1, steps):
        dt = 1  # intervalo de tempo
        random_shock = np.random.normal(0, 1)
        price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shock)
        prices.append(round(float(price), 2))
    return prices

def generate_shared_market_data():
    stock = "Apple"
    price_history = generate_gbm_price_history(
        start_price=random.randint(130, 180),
        steps=10,
        mu=0,      
        sigma=0.1      
    )
    quantity = random.randint(100, 1000)
    previous_quantity = random.randint(10, 100)
    trade_type = random.choice(["limit", "market"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "stock": stock,
        "price_history": price_history,
        "quantity": quantity,
        "previous_quantity": previous_quantity,
        "trade_type": trade_type,
        "timestamp": timestamp
    }

def calculate_word_overlap(text1, text2):
    words1 = set((text1 or "").lower().split())
    words2 = set((text2 or "").lower().split())
    intersection = words1 & words2
    total = words1 | words2
    return round(len(intersection) / len(total) * 100, 2) if total else 0.0

class Investor(Agent):
    def __init__(self, unique_id, model, profile):
        super().__init__(unique_id, model)
        self.profile = profile
        self.state = "neutral"
        self.llama_response = None
        self.mistral_response = None
        self.market_info = {}

    def phase1(self):
        data = self.model.shared_market_data
        text = (
            f"Stock: {data['stock']}\n"
            f"Price History: {data['price_history']}\n"
            f"Order Quantity: {data['quantity']}\n"
            f"Trade Type: {data['trade_type']}\n"
            f"Timestamp: {data['timestamp']}\n"
            f"Investor Profile: {self.profile}\n"
            f"Investor previously bought this stock at: {data['price_history'][0]}\n"
            f"Investor previously bought {data['previous_quantity']} shares.\n"
            f"Current market price (last value): {data['price_history'][-1]}"
        )
        self.model.questions[self.unique_id] = text
        self.market_info = data.copy()
        print(f"[Investor {self.unique_id} - {self.profile}] Info sent:\n{text}")

    def phase2(self): 
        pass

    def phase3(self):
        r1 = self.model.llama_responses.get(self.unique_id)
        r2 = self.model.mistral_responses.get(self.unique_id)
        if r1: self.llama_response = r1
        if r2: self.mistral_response = r2
        if r1: print(f"[Investor {self.unique_id}] Analyst 1: \"{r1}\"", end=" ")
        if r2: print(f"[Investor {self.unique_id}] Analyst 2: \"{r2}\"", end=" ")
        if all("INVEST" in (r or "").upper() for r in (r1, r2)) and all("DO NOT INVEST" not in (r or "").upper() for r in (r1, r2)):
            self.state = "invested"
        else:
            self.state = "did not invest"
        print(f"→ Decision: {self.state}")
        if r1 and r2:
            print(f"→ Overlap: {calculate_word_overlap(r1, r2)}%")
        self.model.llama_responses[self.unique_id] = None
        self.model.mistral_responses[self.unique_id] = None

class AnalystLlama(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def phase1(self): 
        pass

    def phase2(self):
        for agent_id, question in self.model.questions.items():
            if question:
                profile = self.model.schedule.agents[agent_id].profile
                prompt = (
                    f"{question}\n\n"
                    "Given the investor profile and historical price trend, decide which action is best:\n"
                    "INVEST: Buy more\n"
                    "HOLD: Do nothing\n"
                    "SELL: Exit position\n\n"
                    "Respond with one of INVEST, HOLD or SELL, followed by a short reason (max 5 words).\n"
                    "Examples:\n"
                    "INVEST: price rising steadily\n"
                    "HOLD: price fluctuation uncertain\n"
                    "SELL: price dropping significantly\n"
                    "Your answer:"
                )
                messages = [{"role": "user", "content": prompt}]
                try:
                    response = self.model.llm_llama.create_chat_completion(
                        messages=messages,
                        max_tokens=250,
                        temperature=1.5,
                        top_p=0.95
                    )
                    content = response["choices"][0]["message"]["content"].strip().split("\n")[0]
                    print(f"[Analyst 1 - LLaMA 2] Response to Investor {agent_id}: {content}")
                    self.model.llama_responses[agent_id] = content
                except Exception as e:
                    print(f"[Analyst 1 - LLaMA 2] Error: {e}")

    def phase3(self): 
        pass

class AnalystMistral(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def phase1(self): 
        pass

    def phase2(self):
        for agent_id, question in self.model.questions.items():
            if question:
                profile = self.model.schedule.agents[agent_id].profile
                prompt = (
                    f"{question}\n\n"
                   "Given the investor profile, historical price trend, and current market context, decide one of the following actions:\n"
                    "INVEST: Buy more\n"
                    "HOLD: Do nothing\n"
                    "SELL: Exit position\n"
                    f"Investor is {profile}. Respond with one action followed by a brief reason.\n"
                    "The reason must be short (maximum 5 words) and refer to the stock trend, price change, or investor profile.\n"
                    "Answer in one line only.\n"
                    "Examples:\n"
                    "INVEST: price rising steadily\n"
                    "HOLD: price fluctuation uncertain\n"
                    "SELL: price dropping significantly\n"
                    "Your answer:"
                )
                messages = [{"role": "user", "content": prompt}]
                try:
                    response = self.model.llm_mistral.create_chat_completion(
                        messages=messages,
                        max_tokens=250,
                        temperature=0.8,
                        top_p=0.95
                    )
                    content = response["choices"][0]["message"]["content"].strip().split("\n")[0]
                    print(f"[Analyst 2 - Mistral] Response to Investor {agent_id}: {content}")
                    self.model.mistral_responses[agent_id] = content
                except Exception as e:
                    print(f"[Analyst 2 - Mistral] Error: {e}")

    def phase3(self): 
        pass

class MarketModel(Model):
    def __init__(self):
        self.llm_llama = Llama(model_path="llama-2-7b-chat.Q4_K_M.gguf", n_ctx=1024, chat_format="llama-2")
        self.llm_mistral = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=1024, chat_format="mistral-instruct")
        self.schedule = StagedActivation(self, stage_list=["phase1", "phase2", "phase3"])
        self.questions = {}
        self.llama_responses = {}
        self.mistral_responses = {}
        self.shared_market_data = {}

        for i, profile in enumerate(["conservative", "moderate", "aggressive"]):
            self.schedule.add(Investor(i, self, profile))
        self.schedule.add(AnalystLlama(10, self))
        self.schedule.add(AnalystMistral(11, self))

    def step(self, step_num):
        print(f"\n--- STEP {step_num} ---")
        self.shared_market_data = generate_shared_market_data()
        self.schedule.step()

    def save_results_to_csv(self, step_num):
        write_header = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "step", "investor_id", "profile", "price_history", "previous_quantity", "quantity", 
                "trade_type", "llama_response", "mistral_response", "decision", "overlap", "possible_return"
            ])
            if write_header:
                writer.writeheader()
            for agent in self.schedule.agents:
                if isinstance(agent, Investor):
                    r1 = agent.llama_response or ""
                    r2 = agent.mistral_response or ""
                    overlap = calculate_word_overlap(r1, r2)

                    price_history = [float(x) for x in agent.market_info.get("price_history", [])]
                    previous_qty = agent.market_info.get("previous_quantity", 0)
                    if price_history and len(price_history) >= 2:
                        possible_return = round((price_history[-1] - price_history[0]) * previous_qty, 2)
                    else:
                        possible_return = 0.0

                    writer.writerow({
                        "step": step_num,
                        "investor_id": agent.unique_id,
                        "profile": agent.profile,
                        "price_history": price_history,
                        "previous_quantity": previous_qty,
                        "quantity": agent.market_info.get("quantity"),
                        "trade_type": agent.market_info.get("trade_type"),
                        "llama_response": r1,
                        "mistral_response": r2,
                        "decision": agent.state,
                        "overlap": overlap,
                        "possible_return": possible_return
                    })

# Execução principal
model = MarketModel()
start_step = get_last_step()
num_steps = 20

for i in range(start_step, start_step + num_steps):
    model.step(i)
    model.save_results_to_csv(i)
    save_current_step(i)





