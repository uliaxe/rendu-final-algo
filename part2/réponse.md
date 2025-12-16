# Partie 2 : Architecture de l'agent IA

[Voir l'exemple de code Python](exemple_agent_ia.py)

## 3.2 Questions d'architecture

### 1. Choix de l'architecture

* **Architecture Choisie** : **Agent Apprenant (Reinforcement Learning)**.
* **Pourquoi ?** :
  * J'ai choisi cette architecture car elle permet à une IA d'apprendre une tâche complexe par elle-même.
  * Contrairement à un programme classique où il faut coder toutes les règles à la main (ex: "si obstacle, tourner"), l'agent apprend quelles actions sont bonnes ou mauvaises en s'entraînant des milliers de fois.
  * C'est la méthode idéale pour s'adapter à des environnements changeants sans avoir besoin de tout prévoir à l'avance.
* **Fonctionnement (Composants)** :
  * *Réseau de Neurones* : C'est le cerveau qui analyse la situation (l'écran, les capteurs) et propose une action.
  * *Politique d'Exploration* : C'est le mécanisme qui décide parfois de tenter une action au hasard pour découvrir de nouvelles stratégies.

### 2. Problématique d'apprentissage

* **Le Défi** : **Trouver l'équilibre entre la prudence et la découverte**.
* **Explication** :
  * Au début, l'IA ne connaît rien de son environnement. Elle doit agir au hasard pour comprendre les règles (physique, objectifs).
  * Le problème est qu'en explorant au hasard, elle risque souvent l'échec critique (perdre la partie, casser le robot) avant de trouver une récompense.
  * Si je la bride trop, elle ne découvre rien. Si je la laisse faire n'importe quoi, elle échoue tout de suite. Il faut donc ajuster progressivement sa liberté d'action.

### 3. Intégration avec l'algorithme génétique

* **Méthode** : **Démarrer avec un avantage**.
* **Principe** :
  * L'Algorithme Génétique est excellent pour trouver rapidement une "configuration de base" viable (une structure qui tient debout).
  * L'Agent Apprenant (DQN) est excellent pour devenir un expert par la pratique, mais il est très lent à démarrer s'il ne sait absolument rien.
* **Ma Solution** :
  1. Je lance d'abord l'Algorithme Génétique pour créer un agent qui possède déjà des réflexes de survie de base.
  2. Je transfère ce "cerveau" pré-entraîné à l'Agent Apprenant.
  3. L'Agent commence donc son entraînement intensif avec de bonnes bases, ce qui lui permet d'atteindre un niveau de performance élevé beaucoup plus vite.
  