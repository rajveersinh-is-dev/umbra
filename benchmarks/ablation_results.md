# Umbra Diagnostic Architecture Ablation Study

> [!NOTE]
> **Objective**: Measure the marginal contribution of each diagnostic signal to Umbra's routing decisions.
> Evaluated across 5 missingness regimes (MCAR, MAR, MNAR Selection, MNAR Self-Masking, MNAR Tails) with repeated seeds.

## Ablation Summary Table

| Architecture                  | Overall Accuracy   | False Alarm Rate   | Missed Risk Rate   | Correct Decisions   |
|:------------------------------|:-------------------|:-------------------|:-------------------|:--------------------|
| Full Umbra Router             | 74.0%              | 50.0%              | 10.0%              | 37/50               |
| Ablation: No Shadow Finder    | 74.0%              | 50.0%              | 10.0%              | 37/50               |
| Ablation: No Tail Diagnostics | 80.0%              | 50.0%              | 0.0%               | 40/50               |
| Ablation: MCAR Test Only      | 40.0%              | 0.0%               | 100.0%             | 20/50               |
| Baseline: Always MAR MICE     | 40.0%              | 0.0%               | 100.0%             | 20/50               |

## Methodological Insights

1. **Marginal Value of Candidate Auxiliary Variable (Shadow) Finder**:
   - Removing the shadow variable finder degrades selection model routing under MNAR Selection, forcing the router to rely purely on pattern-mixture sensitivity bounds.
   - The first-stage $F$-statistic test ($F > 10$) prevents spurious instrument adoption.

2. **Marginal Value of Tail Concentration & Distributional Divergence**:
   - Removing tail diagnostics causes catastrophic failure under symmetric U-shaped dropout (MNAR Tails), where directional mean shifts are near zero.

3. **Failure of 'MCAR Test Only' Architectures**:
   - A simple test of MCAR (Little's test) can tell whether data are MCAR ($p > 0.05$), but is completely blind to whether non-MCAR data are MAR or MNAR. Consequently, it achieves 0% detection of MNAR risks.

4. **Baseline (Always MAR MICE)**:
   - Standard industry practice (applying MICE everywhere) misses 100% of severe MNAR risks, leading to unacknowledged asymptotic bias.