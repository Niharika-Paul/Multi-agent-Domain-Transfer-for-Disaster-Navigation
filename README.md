# Domain Transfer of Swarm of Agents in Disaster Navigation

> A decentralized multi-agent framework for adaptive disaster navigation in dynamic urban environments using swarm intelligence and spiking-inspired decision mechanisms.

---

## Overview

This project presents a **swarm-based decentralized system** where a population of agents collaboratively performs mapping, risk assessment, resource allocation, and routing under partial and evolving information.

Instead of relying on a fixed set of agents or a central controller, the system consists of **multiple interacting agents (a swarm)** that operate independently while sharing local information. Through these interactions, coordinated global behavior emerges.

The system demonstrates **domain transfer**, where agents generalize their behavior across different disaster scenarios without retraining.

---

## Key Features

- **Decentralized swarm intelligence** without central control
- **Emergent coordination** from multiple interacting agents
- **Domain transfer** across different disaster scenarios
- **Real-world urban dataset** (Tokyo) with population modeling
- **Spiking-inspired adaptive decision mechanism**
- **Dynamic disaster simulation** with evolving hazards
- **Comparative evaluation** with A\* and Dijkstra algorithms

---

## System Architecture

The system is composed of a **swarm of agents**, where each agent operates autonomously but contributes to a collective objective.

Rather than fixed roles, agents exhibit **functional specialization through behavior**, including:

| Behavior | Description |
|---|---|
| Exploration & Mapping | Agents survey and build a shared map of the environment |
| Hazard Detection | Agents estimate risk levels across zones |
| Resource-Aware Decision Making | Agents factor in available resources when planning |
| Safe Path Generation | Agents compute and share viable navigation routes |

### Coordination Mechanisms

Agents coordinate through:

- **Local observations** — each agent perceives its immediate environment
- **Shared memory structures** — global knowledge built from individual contributions
- **Indirect communication** — pheromone-like signaling for emergent guidance

This design enables **scalable and robust coordination**, even when agents have incomplete or outdated information.

---

## Methodology

```
1. Construction of a multi-layer urban graph
      │
      ▼
2. Dynamic disaster injection and environment updates
      │
      ▼
3. Deployment of a swarm of agents with decentralized decision-making
      │
      ▼
4. Spiking-inspired scoring and action selection
      │
      ▼
5. Emergent coordination through local interactions
      │
      ▼
6. Routing and comparison with baseline algorithms (A*, Dijkstra)
```

---

## Results

The swarm-based system achieves **performance comparable to A\*** while offering:

- Improved **adaptability** in dynamic environments
- Greater **robustness** under incomplete information
- Coordination that **emerges from collective agent behavior** rather than centralized planning

---

## Collaborators

- [Niharika Paul](https://github.com/Niharika-Paul)
- [Niharika Saha](https://github.com/niharika-saha)
- [Neha Nair](https://github.com/nehanpnair)
