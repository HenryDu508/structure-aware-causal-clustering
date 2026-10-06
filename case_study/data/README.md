# Sachs et al. (2005) flow cytometry data

Single-cell measurements of 11 phosphorylated proteins and phospholipids in primary human
CD4+ T cells under nine stimulatory or inhibitory conditions. Each file holds one condition
(one row per cell, 7,466 cells in total).

| File | Condition | Cells |
|---|---|---|
| `1. cd3cd28.csv` | anti-CD3/CD28 | 853 |
| `2. cd3cd28icam2.csv` | anti-CD3/CD28 + ICAM-2 | 902 |
| `3. cd3cd28+aktinhib.csv` | anti-CD3/CD28 + Akt inhibitor | 911 |
| `4. cd3cd28+g0076.csv` | anti-CD3/CD28 + G0076 | 723 |
| `5. cd3cd28+psitect.csv` | anti-CD3/CD28 + psitectorigenin | 810 |
| `6. cd3cd28+u0126.csv` | anti-CD3/CD28 + U0126 | 799 |
| `7. cd3cd28+ly.csv` | anti-CD3/CD28 + LY294002 | 848 |
| `8. pma.csv` | PMA | 913 |
| `9. b2camp.csv` | beta2cAMP | 707 |

Columns: `praf`, `pmek`, `plcg`, `PIP2`, `PIP3`, `p44/42` (Erk), `pakts473` (Akt), `PKA`,
`PKC`, `P38`, `pjnk`. The files are CSV exports of the original spreadsheets.

**Source.** Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., and Nolan, G. P. (2005).
Causal protein-signaling networks derived from multiparameter single-cell data.
*Science*, 308(5721), 523-529. https://doi.org/10.1126/science.1105809
The data are distributed as supplementary material of that article.
