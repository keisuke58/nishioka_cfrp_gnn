# Positioning diagram (for paper/slides)

目的: 「どの文脈の研究を参照しているか」を1枚で説明する。

## Diagram (Mermaid)

```mermaid
flowchart LR
  subgraph X["Input source (left → right)"]
    A["Measurements / Sensors / Images\n(IR thermography, DIC, vibration, strain gauges)"] --> B["FE simulation / digital twin fields\n(stress/strain/displacement from FEM)"]
  end

  subgraph Y["Output / Task (bottom → top)"]
    T1["Detection\n(healthy vs damaged / defect vs NDF)"] --> T2["Localization\n(node/region probability)"] --> T3["3D localization\n(layer + planar position)"]
  end

  N["Nishioka et al. 2025\n(FEM + GNN, CFRP perforated,\n3D defect localization)"]:::me
  E["Yehia et al. 2025\n(FE + GNN, subsurface damage,\nnode-level binary)"]:::ref
  W["Wijethunga et al. 2025\n(Dual-graph GNN, SHM sensors,\ndetection + localization)"]:::ref

  A --- W
  B --- E
  B --- N

  T3 --- N
  T1 --- E
  T2 --- W

  classDef me fill:#fff4cc,stroke:#444,stroke-width:1px;
  classDef ref fill:#eef6ff,stroke:#444,stroke-width:1px;
```

## How to use
- slides: この1枚を「Related Work / Positioning」に置く
- paper: Related Work末尾に小図として挿入し、本文で「YehiaはFE×GNNでsubsurface（二値）」「Wijethungaはsensor-graph SHM（dual-graph）」と対比する

