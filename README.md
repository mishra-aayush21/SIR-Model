
---

# SIR Model Simulation

This project simulates disease spread using an **agent-based SIR model** with extensions for **vaccination** and **adaptive behaviors**. Includes a **Pygame visualization**.

---

## Files

1. **`basic_sir.py`**:
   - Basic SIR model without vaccination.
   - Simulates disease spread with infection (`beta`) and recovery (`gamma`) rates.

2. **`sir_vac.py`**:
   - SIR model with vaccination.
   - Vaccinated agents have reduced infection probability and faster recovery.

3. **`sir_adaptivevac.py`**:
   - SIR model with adaptive vaccination and dynamic rewiring.
   - Agents switch to vaccination if infection exceeds a threshold and rewire connections to avoid infected individuals.

4. **`sir_pygame.py`**:
   - Pygame visualization of the SIR model.
   - Displays agents as colored dots (S: Green, I: Red, R: Blue, V: Yellow) with real-time updates.

---

## How to Run

1. Install required libraries:
   ```bash
   pip install numpy matplotlib pygame tqdm
   ```

2. Run the desired script:
   ```bash
   python basic_sir.py          # Basic SIR model
   python sir_vac.py            # SIR model with vaccination
   python sir_adaptivevac.py    # SIR model with adaptive vaccination
   python sir_pygame.py         # Pygame visualization
   ```

3. For `sir_pygame.py`, a window will open showing the simulation. Close the window to stop.

---

## Parameters

Modify parameters in the scripts (e.g., `N`, `beta`, `gamma`, vaccine coverage, etc.) to explore different scenarios.

---
