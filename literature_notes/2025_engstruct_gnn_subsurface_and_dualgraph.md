# Literature notes (GNN × SHM/NDE, 2025)
Generated/maintained in this repo for future reference.

## 1) Yehia et al., Engineering Structures (2025): FE simulations × GNN for subsurface detection
- **Paper**: Ayatollah S. Yehia, Devin K. Harris, Amro Alabsi Aljundi, *“What lies within: Utilizing graph neural networks for subsurface detection in finite element simulations”*, **Engineering Structures**, 341 (2025) 120842. doi:10.1016/j.engstruct.2025.120842
- **Task**: **node-level binary classification** (damaged vs intact) on FE mesh nodes; goal is to infer **subsurface/void damage** using information that is *available on/near the surface* (the paper positions this as aligned with surface-field measurement such as DIC).
- **Core idea**: convert each FE simulation to a graph and use a mesh-oriented GNN (**MeshGraphNet**) to map surface-response features to internal damage labels.

### Method highlights (what is likely transferable)
- **Graph design with “cross edges”**:
  - Uses two node groups (conceptually “internal/intact-side” vs “surface/observed-side”) and introduces **A–B cross edges** linking internal nodes to nearest surface nodes.
  - The paper reports these A–B edges are important (removing them hurts detection), which suggests “surface ↔ interior information routing” is critical for subsurface tasks.
- **Features**:
  - Uses **strain** and **displacement** related features from FE; reports strain tends to be more informative than displacement for damage detection in their setup.
  - Uses geometric/relative-position information (e.g., distances) as edge features (MeshGraphNet style).
- **Training data scale**: reports diminishing returns after ~O(3000) graphs in their study (order-of-magnitude guidance).

### Reported performance (key numbers to remember)
- **Damaged-class F1**: ~0.691 (precision ~0.702, recall ~0.683) on held-out test data (imbalanced setting).
- **Intact-class F1**: high (~0.976) due to dominance of intact nodes.
- **Damage size sensitivity**: smaller damage is harder; larger damage yields better F1.

### How this maps to THIS repo’s pipeline (actionable takeaways)
- **Best “direct transfer” concept**: add an explicit mechanism for **surface→interior routing**.
  - Implementation analog in our codebase: add/augment edges so that interior (layer) nodes receive messages from “most related” surface/near-surface nodes (nearest-neighbor cross edges; layer-adjacent shortcut edges; hole-centric shortcut edges).
- **Two-stage head fits well**: Yehia is effectively “stage-1 detection”; our existing (NDF vs defect) detection head can be treated as the Yehia-like component, while our second head handles defect-type (and layer constraints).
- **Metrics**: replicate their emphasis on **damaged-class F1** (not accuracy) when comparing architectural variants.

---

## 2) Wijethunga et al., Engineering Structures (2025): Dual-graph GNN for SHM damage detection/localization
- **Paper**: Rashinda Wijethunga, Jagath Samarabandu, Ayan Sadhu, *“Robust and efficient dual-graph neural networks for structural damage detection and localization”*, **Engineering Structures**, 343 (2025) 121265. doi:10.1016/j.engstruct.2025.121265
- **Task**: SHM **damage detection and localization** from measurement data, modeling sensors/locations as a graph.
- **Core idea**: **dual-graph** construction:
  - one graph encodes **physical/sensor topology** (geometry/placement),
  - another graph encodes **feature-correlation topology** (relationships derived from the signals),
  then fuses both for robustness.

### How this relates to our project
- Yehia is closer in “FEM mesh + internal damage inference”, while Wijethunga is closer in “sensor-graph SHM”.
- The transferable concept is: **multiple graphs / multiple adjacency views** can help robustness.
  - Our analog: combine (i) mesh adjacency, (ii) kNN in Euclidean space, and (iii) “same-hole” / “same-layer” / “cross-layer” shortcut edges; or run attention over multiple edge sets.

---

## 3) Where this connects to our published work
- **Nishioka et al., Frontiers in Materials (2025)**: integrates FEM + GNN for defect localization in perforated CFRP using stress-distribution features, bridging the “FEM-based inference” side (like Yehia) with SHM/NDE application constraints.

## 4) TODO (optional future experiments in this repo)
- **E1**: Add cross-edges (surface ↔ interior) and ablate (on/off) similar to Yehia’s A–B edge study.
- **E2**: Add edge attributes (relative position, layer delta, hole-centered radial features) and compare to current edge-only GAT.
- **E3**: Evaluate detection head alone as binary task; compare to multi-class end-to-end (align evaluation with Yehia’s damaged-F1).

