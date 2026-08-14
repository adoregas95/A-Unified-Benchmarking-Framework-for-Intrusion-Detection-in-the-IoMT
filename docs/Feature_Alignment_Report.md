# Feature Alignment Report
## Cross-Dataset CICFlowMeter Feature Verification

*Generated: March 2026*
*Script: `preprocessing/feature_alignment_verification.py`*

---

## Summary

All four datasets in the unified benchmarking framework share **76 identical
ML-usable CICFlowMeter features** after column name standardization.

| Dataset | Total Cols | ML Features | Column Style | Alignment |
|---------|-----------|-------------|--------------|-----------|
| CICIoMT2024 | 86 | 76 | Full names (reference) | Reference |
| CIC-BoT-IoT | 85 | 76 | Abbreviated (48 renamed) | 100% Match |
| CIC-IoT-DIAD-2024 | 84 | 76 | Full names (identical) | 100% Match |
| CIC-ToN-IoT | 85 | 76 | Abbreviated (48 renamed) | 100% Match |

---

## Column Categories

Each dataset's columns fall into three categories:

**Flow Identifiers (7 columns — excluded from ML):**
Flow ID, Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp

**Label Columns (varies by dataset — excluded from ML):**
- CICIoMT2024: Label, Class, Sub-Class (3 columns)
- CIC-BoT-IoT: Label, Attack (2 columns)
- CIC-IoT-DIAD-2024: Label (1 column, contains "NeedManualLabel")
- CIC-ToN-IoT: Label, Attack (2 columns)

**ML Features (76 columns — used for training/inference):**
See complete list below.

---

## Column Name Standardization

CIC-BoT-IoT and CIC-ToN-IoT use abbreviated column names (e.g., "Tot Fwd Pkts"
instead of "Total Fwd Packet"). The `data_loader.py` includes an
`ABBREVIATED_TO_FULL` mapping dictionary that renames all 48 differing columns
to match the CICIoMT2024/CIC-IoT-DIAD-2024 standard.

48 of 76 feature names differ between abbreviated and full formats.
28 of 76 feature names are identical across all datasets.

---

## Complete Feature List (76 ML Features)

 1. Flow Duration
 2. Total Fwd Packet
 3. Total Bwd packets
 4. Total Length of Fwd Packet
 5. Total Length of Bwd Packet
 6. Fwd Packet Length Max
 7. Fwd Packet Length Min
 8. Fwd Packet Length Mean
 9. Fwd Packet Length Std
10. Bwd Packet Length Max
11. Bwd Packet Length Min
12. Bwd Packet Length Mean
13. Bwd Packet Length Std
14. Flow Bytes/s
15. Flow Packets/s
16. Flow IAT Mean
17. Flow IAT Std
18. Flow IAT Max
19. Flow IAT Min
20. Fwd IAT Total
21. Fwd IAT Mean
22. Fwd IAT Std
23. Fwd IAT Max
24. Fwd IAT Min
25. Bwd IAT Total
26. Bwd IAT Mean
27. Bwd IAT Std
28. Bwd IAT Max
29. Bwd IAT Min
30. Fwd PSH Flags
31. Bwd PSH Flags
32. Fwd URG Flags
33. Bwd URG Flags
34. Fwd Header Length
35. Bwd Header Length
36. Fwd Packets/s
37. Bwd Packets/s
38. Packet Length Min
39. Packet Length Max
40. Packet Length Mean
41. Packet Length Std
42. Packet Length Variance
43. FIN Flag Count
44. SYN Flag Count
45. RST Flag Count
46. PSH Flag Count
47. ACK Flag Count
48. URG Flag Count
49. CWR Flag Count
50. ECE Flag Count
51. Down/Up Ratio
52. Average Packet Size
53. Fwd Segment Size Avg
54. Bwd Segment Size Avg
55. Fwd Bytes/Bulk Avg
56. Fwd Packet/Bulk Avg
57. Fwd Bulk Rate Avg
58. Bwd Bytes/Bulk Avg
59. Bwd Packet/Bulk Avg
60. Bwd Bulk Rate Avg
61. Subflow Fwd Packets
62. Subflow Fwd Bytes
63. Subflow Bwd Packets
64. Subflow Bwd Bytes
65. FWD Init Win Bytes
66. Bwd Init Win Bytes
67. Fwd Act Data Pkts
68. Fwd Seg Size Min
69. Active Mean
70. Active Std
71. Active Max
72. Active Min
73. Idle Mean
74. Idle Std
75. Idle Max
76. Idle Min

---

## Notes

- CIC-IoT-DIAD-2024 has 5 files with column header mismatches (likely minor
  formatting differences). These should be investigated during preprocessing.
- The `Label` column in CIC-IoT-DIAD-2024 contains "NeedManualLabel" for all
  rows; actual labels are derived from the folder structure (attack family names).
- CIC-BoT-IoT and CIC-ToN-IoT use binary Label (0/1) plus an Attack column
  with family names.

---

## Reproduction

Run the verification script from the project root:

```bash
cd dissertation
python preprocessing/feature_alignment_verification.py
```

Or specify a custom data root:

```bash
python preprocessing/feature_alignment_verification.py --data-root /path/to/data
```
