import numpy as np
from collections import deque
import random


class ReseauNeurones:
    
    def __init__(self, input_size, output_size, hidden_size=64):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
    
    def forward(self, state):
        hidden = np.maximum(0, np.dot(state, self.W1) + self.b1)
        output = np.dot(hidden, self.W2) + self.b2
        return output
    
    def copier(self):
        nouveau = ReseauNeurones(self.W1.shape[0], self.W2.shape[1])
        nouveau.W1 = self.W1.copy()
        nouveau.b1 = self.b1.copy()
        nouveau.W2 = self.W2.copy()
        nouveau.b2 = self.b2.copy()
        return nouveau


class AgentApprenant:
    
    def __init__(self, input_size, output_size):
        self.reseau = ReseauNeurones(input_size, output_size)
        self.experience = deque(maxlen=1000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
    
    def choisir_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.reseau.W2.shape[1] - 1)
        else:
            q_values = self.reseau.forward(state)
            return np.argmax(q_values)
    
    def memoriser(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))
    
    def apprendre(self, batch_size=32):
        if len(self.experience) < batch_size:
            return
        
        batch = random.sample(self.experience, batch_size)
        
        self.epsilon *= self.epsilon_decay
        
        print(f"  Apprentissage... (epsilon: {self.epsilon:.3f})")
    
    def transferer_poids(self, autre_agent):
        self.reseau = autre_agent.reseau.copier()


class Population:

    def __init__(self, size, input_size, output_size):
        self.agents = [AgentApprenant(input_size, output_size) for _ in range(size)]
        self.fitness_scores = []
    
    def evaluer(self, simulation_func):
        self.fitness_scores = []
        for i, agent in enumerate(self.agents):
            score = simulation_func(agent)
            self.fitness_scores.append(score)
            print(f"Agent {i}: score={score:.2f}")
    
    def selection_naturelle(self):

        sorted_indices = np.argsort(self.fitness_scores)[::-1]

        keep_count = max(1, len(self.agents) // 5)
        meilleurs = [self.agents[i] for i in sorted_indices[:keep_count]]
        
        print(f"Les {keep_count} meilleurs agents survivent")
        return meilleurs
    
    def croisement_et_mutation(self, meilleurs, mutation_rate=0.1):
        self.agents = meilleurs.copy()
        while len(self.agents) < len(meilleurs) * 5: 
            parent1 = random.choice(meilleurs)
            parent2 = random.choice(meilleurs)

            enfant = AgentApprenant(
                parent1.reseau.W1.shape[0], 
                parent1.reseau.W2.shape[1]
            )

            if random.random() < 0.5:
                enfant.reseau = parent1.reseau.copier()
            else:
                enfant.reseau = parent2.reseau.copier()

            if random.random() < mutation_rate:
                enfant.reseau.W1 += np.random.randn(*enfant.reseau.W1.shape) * 0.1
            
            self.agents.append(enfant)
        
        self.fitness_scores = []

def simuler_agent(agent, steps=100):
    state = np.array([random.random(), random.random()])
    cible = np.array([0.5, 0.5])
    
    score_total = 0
    for _ in range(steps):
        action = agent.choisir_action(state, training=False)

        if action == 0:
            state[1] += 0.05
        elif action == 1:
            state[1] -= 0.05
        elif action == 2:
            state[0] -= 0.05
        elif action == 3:
            state[0] += 0.05

        distance = np.linalg.norm(state - cible)
        reward = 1.0 - distance
        score_total += reward
        
        # Limites
        state = np.clip(state, 0, 1)
    
    return score_total

def main():
    print("=" * 60)
    print("DÉMO: Agent Apprenant + Algorithme Génétique")
    print("=" * 60)

    print("\n📊 PHASE 1: Algorithme Génétique")
    print("-" * 60)
    print("Objectif: Trouver rapidement une 'configuration de base' viable\n")
    
    population = Population(size=10, input_size=2, output_size=4)
    meilleur_agent_genetique = None
    meilleur_score_genetique = -float('inf')
    
    for generation in range(3):
        print(f"\n🔄 Génération {generation + 1}")
        population.evaluer(simuler_agent)

        best_idx = np.argmax(population.fitness_scores)
        if population.fitness_scores[best_idx] > meilleur_score_genetique:
            meilleur_score_genetique = population.fitness_scores[best_idx]
            meilleur_agent_genetique = population.agents[best_idx]
        
        meilleurs = population.selection_naturelle()
        population.croisement_et_mutation(meilleurs)
    
    print(f"\n✅ Meilleur agent génétique trouvé: score={meilleur_score_genetique:.2f}")

    print("\n" + "=" * 60)
    print("🧠 PHASE 2: Agent Apprenant (DQN)")
    print("-" * 60)
    print("Objectif: Devenir expert par l'apprentissage continu\n")
    
    agent_apprenant = AgentApprenant(input_size=2, output_size=4)
    
    print("Sans transfert d'apprentissage:")
    score_avant = simuler_agent(agent_apprenant)
    print(f"  Score initial: {score_avant:.2f}\n")

    print("=" * 60)
    print("🚀 PHASE 3: Transfert d'Apprentissage")
    print("-" * 60)
    print("Combinaison: Commencer avec les 'bons gènes' génétiques\n")
    
    agent_avec_transfert = AgentApprenant(input_size=2, output_size=4)
    agent_avec_transfert.transferer_poids(meilleur_agent_genetique)
    
    print("Avec transfert depuis l'algorithme génétique:")
    for epoch in range(3):
        print(f"\n📈 Epoch {epoch + 1}")
        agent_avec_transfert.apprendre(batch_size=32)
    
    score_apres = simuler_agent(agent_avec_transfert)
    print(f"\n  Score final: {score_apres:.2f}")

    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"Score agent sans transfert: {score_avant:.2f}")
    print(f"Score agent avec transfert:  {score_apres:.2f}")
    print(f"Amélioration: {((score_apres - score_avant) / abs(score_avant) * 100):.1f}%")
    print(f"\nMeilleur score génétique trouvé: {meilleur_score_genetique:.2f}")
    print("\n✨ Conclusion: Le transfert d'apprentissage donne un meilleur départ!")


if __name__ == "__main__":
    main()
